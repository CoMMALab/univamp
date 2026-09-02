"""Install a cudaMallocManaged-backed allocator into PyTorch.

On Jetson Thor (UMA) this makes every cuRobo CUDA tensor allocation managed, so a CPU
producer (VAMP) and GPU consumer (cuRobo) share pages without explicit copies.

Usage (must be called BEFORE any CUDA tensor is allocated in the process):

    import univamp
    univamp.install_managed_allocator()
    import torch
    x = torch.zeros(4, device="cuda")   # backed by cudaMallocManaged

Tunables via env vars (read once at first allocation):
    UNIVAMP_ADVISE_GPU=1   set preferred location + accessedBy = device
    UNIVAMP_PREFETCH=1     prefetch to device on the alloc stream
    UNIVAMP_ALLOC_VERBOSE=0
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import lru_cache

_LIB_NAME = "libunivamp_managed.so"
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _LIB_NAME)


@lru_cache(maxsize=1)
def _stats_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_LIB_PATH)
    for fn in (
        "univamp_alloc_count", "univamp_free_count", "univamp_cache_hits",
        "univamp_cache_miss", "univamp_capture_miss", "univamp_bytes_live",
        "univamp_bytes_peak", "univamp_bytes_total", "univamp_bytes_phys",
        "univamp_prefetch_count",
    ):
        getattr(lib, fn).restype = ctypes.c_ulonglong
        getattr(lib, fn).argtypes = []
    for fn in ("univamp_reset_stats", "univamp_empty_cache"):
        getattr(lib, fn).restype = None
        getattr(lib, fn).argtypes = []
    return lib


@lru_cache(maxsize=1)
def managed_allocator():
    """Return the torch CUDAPluggableAllocator backed by cudaMallocManaged."""
    import torch  # local import: allocator must be set before torch CUDA init anyway

    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(
            f"{_LIB_PATH} not found. Build it with `bash univamp/build.sh`."
        )
    return torch.cuda.memory.CUDAPluggableAllocator(
        _LIB_PATH, "univamp_malloc", "univamp_free"
    )


def install_managed_allocator() -> None:
    """Make PyTorch route all CUDA allocations through the managed allocator.

    Raises RuntimeError if CUDA is already initialized (allocator can't be swapped then).
    """
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA already initialized; install_managed_allocator() must run before any "
            "CUDA tensor is created."
        )
    torch.cuda.memory.change_current_allocator(managed_allocator())


@dataclass
class AllocStats:
    alloc_count: int
    free_count: int
    cache_hits: int
    cache_miss: int
    capture_miss: int
    bytes_live: int
    bytes_peak: int
    bytes_total: int
    bytes_phys: int
    prefetch_count: int

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.alloc_count if self.alloc_count else 0.0

    @property
    def mb_peak(self) -> float:
        return self.bytes_peak / (1024 ** 2)

    @property
    def mb_total(self) -> float:
        return self.bytes_total / (1024 ** 2)

    @property
    def mb_phys(self) -> float:
        return self.bytes_phys / (1024 ** 2)

    def summary(self) -> str:
        return (f"allocs={self.alloc_count} hits={self.cache_hits} miss={self.cache_miss} "
                f"hit_rate={self.hit_rate:.4f} capture_miss={self.capture_miss} "
                f"prefetch={self.prefetch_count} peakMB={self.mb_peak:.1f} "
                f"physMB={self.mb_phys:.1f} logicalMB={self.mb_total:.1f}")


def get_alloc_stats() -> AllocStats:
    lib = _stats_lib()
    return AllocStats(
        alloc_count=lib.univamp_alloc_count(),
        free_count=lib.univamp_free_count(),
        cache_hits=lib.univamp_cache_hits(),
        cache_miss=lib.univamp_cache_miss(),
        capture_miss=lib.univamp_capture_miss(),
        bytes_live=lib.univamp_bytes_live(),
        bytes_peak=lib.univamp_bytes_peak(),
        bytes_total=lib.univamp_bytes_total(),
        bytes_phys=lib.univamp_bytes_phys(),
        prefetch_count=lib.univamp_prefetch_count(),
    )


def reset_alloc_stats() -> None:
    _stats_lib().univamp_reset_stats()


def empty_cache() -> None:
    """Physically cudaFree all pooled blocks (call when idle, never during graph capture)."""
    _stats_lib().univamp_empty_cache()


# --- surgical managed allocation (no global swap) --------------------------------------
@lru_cache(maxsize=1)
def managed_mem_pool():
    """A torch MemPool whose allocations use cudaMallocManaged. Use to make ONLY specific
    tensors managed (e.g. the VAMP->cuRobo seed buffer) while cuRobo keeps the stock
    caching allocator + CUDA graphs."""
    import torch
    return torch.cuda.MemPool(managed_allocator().allocator())


def managed_empty(*shape, dtype=None, device="cuda"):
    """Allocate a managed CUDA tensor (zero global-allocator impact). The backing memory is
    cudaMallocManaged so a CPU producer and GPU consumer share it without explicit copies."""
    import torch
    if dtype is None:
        dtype = torch.float32
    with torch.cuda.use_mem_pool(managed_mem_pool()):
        return torch.empty(*shape, dtype=dtype, device=device)
