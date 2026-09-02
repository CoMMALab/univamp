// UniVAMP merged seed-expansion kernel (Jetson Thor UMA).
//
// VAMP (CPU) writes raw RRTC waypoints into a *managed* buffer; this kernel reads them
// zero-copy on the GPU and expands each seed to the TrajOpt action horizon by arc-length
// linear resampling -- the GPU-side equivalent of seeder._interp_to_horizon. The expanded
// seed (B*H*dof) is materialized directly in a device tensor and never crosses the bus or
// touches host memory; only the raw waypoints (Sum M_i * dof, much smaller) are produced by
// the CPU. The kernel runs on a caller stream *before* TrajOpt's CUDA-graph capture and
// allocates nothing, so cuRobo's graphs are untouched.
//
// DOF-agnostic and large-batch ready (mobile-manipulator / quadruped / humanoid endgame):
// dof, horizon and batch are all runtime args; grid is sized for B*H threads.
//
// Build: bash univamp/build.sh  ->  libunivamp_seedexpand.so

#include <cuda_runtime.h>
#include <cstdio>

namespace {

// Binary search: largest index k in [lo, hi-1] with cum[k] <= q. Assumes cum ascending.
__device__ inline int upper_segment(const float* cum, int lo, int hi, float q) {
  // returns segment start index s in [lo, hi-2] such that cum[s] <= q <= cum[s+1]
  int n = hi - lo;                 // number of waypoints for this seed
  if (n <= 1) return lo;           // degenerate; caller handles repeat
  // clamp q into [cum[lo], cum[hi-1]]
  if (q <= cum[lo]) return lo;
  if (q >= cum[hi - 1]) return hi - 2;
  int a = lo, b = hi - 1;          // find k = floor: cum[k] <= q < cum[k+1]
  while (b - a > 1) {
    int m = (a + b) >> 1;
    if (cum[m] <= q) a = m; else b = m;
  }
  return a;
}

// One thread per (b, h). dof-loop inside (dof small relative to B*H).
__global__ void seed_expand_kernel(const float* __restrict__ wp,       // (total_wp, dof)
                                   const int*   __restrict__ seed_off,  // (B+1)
                                   const float* __restrict__ cum,       // (total_wp)
                                   float*       __restrict__ out,       // (B, H, dof)
                                   int B, int H, int dof) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)B * H;
  if (tid >= total) return;
  int b = (int)(tid / H);
  int h = (int)(tid - (long)b * H);

  int lo = seed_off[b];
  int hi = seed_off[b + 1];
  int n  = hi - lo;
  float* o = out + ((long)b * H + h) * dof;

  if (n <= 0) {                    // no waypoints: emit zeros (defensive)
    for (int j = 0; j < dof; ++j) o[j] = 0.0f;
    return;
  }
  if (n == 1) {                    // single waypoint: repeat it across the horizon
    const float* w = wp + (long)lo * dof;
    for (int j = 0; j < dof; ++j) o[j] = w[j];
    return;
  }

  float total_len = cum[hi - 1];   // cum is per-seed cumulative arclength, cum[lo]==0
  float q = (H > 1) ? (total_len * (float)h / (float)(H - 1)) : 0.0f;
  int s = upper_segment(cum, lo, hi, q);   // segment [s, s+1]
  float c0 = cum[s], c1 = cum[s + 1];
  float t = (c1 > c0) ? (q - c0) / (c1 - c0) : 0.0f;

  const float* w0 = wp + (long)s * dof;
  const float* w1 = wp + (long)(s + 1) * dof;
  for (int j = 0; j < dof; ++j) o[j] = w0[j] + t * (w1[j] - w0[j]);
}

}  // namespace

extern "C" {

// Launch the expansion. All pointers are device-accessible (managed inputs, device output).
// Returns 0 on success, else a cudaError_t code (kernel launch error).
int univamp_seed_expand(const float* wp, const int* seed_off, const float* cum,
                        float* out, int B, int H, int dof, cudaStream_t stream) {
  if (B <= 0 || H <= 0 || dof <= 0) return cudaErrorInvalidValue;
  const int threads = 256;
  long total = (long)B * H;
  long blocks = (total + threads - 1) / threads;
  seed_expand_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
      wp, seed_off, cum, out, B, H, dof);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[univamp] seed_expand launch failed: %s\n", cudaGetErrorString(err));
    return (int)err;
  }
  return 0;
}

}  // extern "C"
