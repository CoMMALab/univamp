"""Merged seed-expansion handoff: raw VAMP waypoints (managed) -> expanded device seed (GPU).

This is the zero-copy half of the UMA story. VAMP-RRTC produces variable-length raw waypoint
paths on the CPU. Instead of interpolating them to the TrajOpt horizon on the CPU and copying the
(large) expanded ``(B,H,dof)`` seed across the bus, we:

  1. write only the *raw* waypoints (Sum M_i * dof, much smaller) into a managed buffer that the
     CPU producer fills host-side (true UMA write -- no explicit H2D copy of this payload), and
  2. launch a single fused CUDA kernel (``libunivamp_seedexpand.so``) that reads those managed
     pages zero-copy on the GPU and writes the horizon-expanded seed straight into a device
     tensor that cuRobo TrajOpt consumes.

The expanded seed therefore never touches host memory, and the CPU-side interpolation is moved
onto the otherwise-idle GPU. The kernel allocates nothing and runs before TrajOpt's CUDA-graph
capture, so cuRobo keeps its stock caching allocator + CUDA graphs.

DOF-agnostic and large-batch ready: ``dof`` and ``horizon`` are runtime args; the managed
waypoint buffers come from the surgical ``managed_empty`` MemPool (graphs untouched).
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from typing import List, Sequence

import numpy as np

from univamp.managed_allocator import managed_empty

_LIB_NAME = "libunivamp_seedexpand.so"
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _LIB_NAME)


@lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(
            f"{_LIB_PATH} not found. Build it with `bash univamp/build.sh`."
        )
    lib = ctypes.CDLL(_LIB_PATH)
    # int univamp_seed_expand(wp, seed_off, cum, out, B, H, dof, stream)
    lib.univamp_seed_expand.restype = ctypes.c_int
    lib.univamp_seed_expand.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    return lib


def _host_view(tensor, np_dtype) -> np.ndarray:
    """Return a numpy view aliasing a managed CUDA tensor's memory for host-side writes.

    Valid only for cudaMallocManaged-backed tensors (host-accessible). Writing through this view
    *is* the CPU->GPU handoff under UMA (page ownership flip), with no explicit copy kernel.
    """
    n = tensor.numel()
    itemsize = np.dtype(np_dtype).itemsize
    buf = (ctypes.c_char * (n * itemsize)).from_address(tensor.data_ptr())
    return np.frombuffer(buf, dtype=np_dtype, count=n)


def pack_waypoints(seeds: Sequence[np.ndarray], dof: int, device="cuda"):
    """Pack a list of raw ``(M_i, dof)`` waypoint arrays into managed CSR buffers.

    Returns ``(wp, seed_off, cum, B)`` where ``wp`` is the managed ``(total_wp, dof)`` waypoint
    tensor, ``seed_off`` the managed ``(B+1)`` int32 CSR offsets, ``cum`` the managed
    ``(total_wp,)`` per-seed cumulative arc length (cum[start]==0). The CPU fills all three via
    host views (the UMA write); no torch copy kernel runs.
    """
    B = len(seeds)
    counts = np.fromiter((int(s.shape[0]) for s in seeds), dtype=np.int64, count=B)
    offsets = np.zeros(B + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    total_wp = int(offsets[-1])

    wp = managed_empty(total_wp, dof, dtype=_torch_f32(), device=device)
    seed_off = managed_empty(B + 1, dtype=_torch_i32(), device=device)
    cum = managed_empty(total_wp, dtype=_torch_f32(), device=device)

    wp_h = _host_view(wp, np.float32).reshape(total_wp, dof)
    off_h = _host_view(seed_off, np.int32)
    cum_h = _host_view(cum, np.float32)

    off_h[:] = offsets
    for i, s in enumerate(seeds):
        lo, hi = offsets[i], offsets[i + 1]
        sa = np.ascontiguousarray(s[:, :dof], dtype=np.float32)
        wp_h[lo:hi] = sa
        if hi - lo <= 1:
            cum_h[lo:hi] = 0.0
        else:
            seg = np.linalg.norm(np.diff(sa, axis=0), axis=1)
            cum_h[lo:hi] = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    return wp, seed_off, cum, B


def expand_seeds_uma(seeds: List[np.ndarray], horizon: int, dof: int, device="cuda"):
    """Zero-copy merged handoff: raw waypoints -> ``(1, B, horizon, dof)`` device seed tensor.

    ``seeds`` is the list of raw VAMP paths (each ``(M_i, dof)``). The returned tensor is a
    **stock-allocator** device tensor (so it can be fed to TrajOpt without disturbing graphs).
    """
    import torch

    if not seeds:
        return None
    wp, seed_off, cum, B = pack_waypoints(seeds, dof, device=device)
    out = torch.empty(B, horizon, dof, dtype=torch.float32, device=device)  # stock allocator
    stream = torch.cuda.current_stream().cuda_stream
    rc = _lib().univamp_seed_expand(
        ctypes.c_void_p(wp.data_ptr()), ctypes.c_void_p(seed_off.data_ptr()),
        ctypes.c_void_p(cum.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(B), ctypes.c_int(horizon), ctypes.c_int(dof),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"univamp_seed_expand failed (cudaError={rc})")
    return out.unsqueeze(0)  # (1, B, horizon, dof) to match _get_graph_seed_trajectories


@lru_cache(maxsize=1)
def _torch_f32():
    import torch
    return torch.float32


@lru_cache(maxsize=1)
def _torch_i32():
    import torch
    return torch.int32
