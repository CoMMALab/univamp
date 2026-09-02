// UniVAMP caching managed-memory allocator for PyTorch (Jetson Thor UMA).
//
// Exposes C symbols compatible with torch.cuda.memory.CUDAPluggableAllocator. Backs every
// CUDA tensor with cudaMallocManaged, but POOLS blocks for reuse (like torch's caching
// allocator) so we avoid per-tensor cudaMalloc/cudaFree churn and make allocation
// CUDA-graph-capture safe:
//   * Physical cudaMallocManaged + memAdvise + prefetch happen only on a pool MISS while
//     NOT capturing a CUDA graph.
//   * During graph capture, requests are served from the pool (cuRobo warms up before
//     capture, so the pool is already populated) -- no illegal in-capture allocation.
//   * free() returns the block to the pool instead of calling cudaFree.
//
// Stats expose cache hit rate (key UMA success metric) and physical vs logical bytes.
//
// Build: bash univamp/build.sh  ->  libunivamp_managed.so

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <map>
#include <unordered_map>
#include <vector>

namespace {
// --- tunables (env) --------------------------------------------------------------------
bool g_advise_gpu  = true;   // UNIVAMP_ADVISE_GPU
bool g_prefetch    = true;   // UNIVAMP_PREFETCH
bool g_verbose     = false;  // UNIVAMP_ALLOC_VERBOSE
bool g_caching     = true;   // UNIVAMP_CACHING (1=pool reuse, 0=passthrough cudaFree)
int  g_prestock    = 2;      // UNIVAMP_PRESTOCK: extra spare blocks per size-class on a miss,
                             // so CUDA-graph capture (needs +1 live block) hits the pool
                             // instead of doing an illegal in-capture cudaMallocManaged.
bool g_initialized = false;

constexpr size_t kAlign = 512;  // round allocations up to this granularity for pooling

std::mutex g_mtx;
std::map<size_t, std::vector<void*>>     g_free;   // size-class -> free blocks
std::unordered_map<void*, size_t>        g_size;   // ptr -> physical block size

// --- instrumentation -------------------------------------------------------------------
std::atomic<unsigned long long> g_alloc_count{0};    // logical mallocs
std::atomic<unsigned long long> g_free_count{0};
std::atomic<unsigned long long> g_cache_hits{0};     // served from pool
std::atomic<unsigned long long> g_cache_miss{0};     // physical cudaMallocManaged
std::atomic<unsigned long long> g_bytes_live{0};     // logical live
std::atomic<unsigned long long> g_bytes_peak{0};
std::atomic<unsigned long long> g_bytes_total{0};    // logical cumulative
std::atomic<unsigned long long> g_bytes_phys{0};     // physical cumulative (cudaMallocManaged)
std::atomic<unsigned long long> g_prefetch_count{0};
std::atomic<unsigned long long> g_capture_miss{0};   // misses forced during capture (risky)

bool env_bool(const char* n, bool d) {
  const char* v = std::getenv(n);
  if (!v) return d;
  return !(v[0]=='0'||v[0]=='f'||v[0]=='F'||v[0]=='n'||v[0]=='N');
}
void ensure_init() {
  if (g_initialized) return;
  g_advise_gpu = env_bool("UNIVAMP_ADVISE_GPU", true);
  g_prefetch   = env_bool("UNIVAMP_PREFETCH", true);
  g_verbose    = env_bool("UNIVAMP_ALLOC_VERBOSE", false);
  g_caching    = env_bool("UNIVAMP_CACHING", true);
  if (const char* p = std::getenv("UNIVAMP_PRESTOCK")) g_prestock = std::atoi(p);
  g_initialized = true;
}
inline size_t round_up(size_t s) { s = s ? s : 1; return ((s + kAlign - 1) / kAlign) * kAlign; }

bool stream_capturing(cudaStream_t stream) {
  if (!stream) return false;
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess)
    return cap != cudaStreamCaptureStatusNone;
  return false;
}

void apply_hints(void* ptr, size_t size, int device, cudaStream_t stream, bool capturing) {
  if (capturing || size == 0) return;  // advise/prefetch illegal during capture
  cudaMemLocation loc; loc.type = cudaMemLocationTypeDevice; loc.id = device;
  if (g_advise_gpu) {
    cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, loc);
    cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, loc);
  }
  if (g_prefetch && cudaMemPrefetchAsync(ptr, size, loc, 0, stream) == cudaSuccess)
    g_prefetch_count.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

extern "C" {

void* univamp_malloc(size_t size, int device, cudaStream_t stream) {
  ensure_init();
  const size_t rounded = round_up(size);
  const bool capturing = stream_capturing(stream);
  void* ptr = nullptr;

  {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_free.find(rounded);
    if (g_caching && it != g_free.end() && !it->second.empty()) {
      ptr = it->second.back();
      it->second.pop_back();
      g_cache_hits.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (!ptr) {
    // Pool miss: must physically allocate. Illegal during capture, but we try as a last
    // resort (and count it) so failures are visible rather than silent corruption.
    if (capturing) g_capture_miss.fetch_add(1, std::memory_order_relaxed);
    cudaError_t err = cudaMallocManaged(&ptr, rounded, cudaMemAttachGlobal);
    if (err != cudaSuccess || !ptr) {
      fprintf(stderr, "[univamp] cudaMallocManaged(%zu) failed: %s%s\n", rounded,
              cudaGetErrorString(err), capturing ? " (during graph capture)" : "");
      return nullptr;
    }
    g_cache_miss.fetch_add(1, std::memory_order_relaxed);
    g_bytes_phys.fetch_add(rounded, std::memory_order_relaxed);
    apply_hints(ptr, rounded, device, stream, capturing);
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      g_size[ptr] = rounded;
    }
    // Over-provision: pre-allocate spare blocks of this size into the pool so a later
    // CUDA-graph capture (which needs an extra concurrent block) hits the cache instead
    // of attempting an illegal in-capture cudaMallocManaged. Only when not capturing.
    if (g_caching && !capturing && g_prestock > 0) {
      for (int i = 0; i < g_prestock; ++i) {
        void* sp = nullptr;
        if (cudaMallocManaged(&sp, rounded, cudaMemAttachGlobal) == cudaSuccess && sp) {
          g_bytes_phys.fetch_add(rounded, std::memory_order_relaxed);
          apply_hints(sp, rounded, device, stream, false);
          std::lock_guard<std::mutex> lk(g_mtx);
          g_size[sp] = rounded;
          g_free[rounded].push_back(sp);
        }
      }
    }
  } else if (g_prefetch && !capturing && size > 0) {
    // Cache hit: cheap re-home to GPU (no advise, which is sticky from first alloc).
    cudaMemLocation loc; loc.type = cudaMemLocationTypeDevice; loc.id = device;
    if (cudaMemPrefetchAsync(ptr, rounded, loc, 0, stream) == cudaSuccess)
      g_prefetch_count.fetch_add(1, std::memory_order_relaxed);
  }

  unsigned long long live = g_bytes_live.fetch_add(rounded, std::memory_order_relaxed) + rounded;
  unsigned long long peak = g_bytes_peak.load(std::memory_order_relaxed);
  while (live > peak && !g_bytes_peak.compare_exchange_weak(peak, live)) {}
  g_bytes_total.fetch_add(rounded, std::memory_order_relaxed);
  g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  if (g_verbose) fprintf(stderr, "[univamp] malloc %p size=%zu(%zu) dev=%d cap=%d\n",
                         ptr, size, rounded, device, capturing);
  return ptr;
}

void univamp_free(void* ptr, size_t size, int device, cudaStream_t stream) {
  (void)device; (void)stream;
  if (!ptr) return;
  ensure_init();
  size_t rounded;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_size.find(ptr);
    rounded = (it != g_size.end()) ? it->second : round_up(size);
    if (g_caching) {
      g_free[rounded].push_back(ptr);   // return to pool; do NOT cudaFree
    } else {
      g_size.erase(ptr);
    }
  }
  if (!g_caching) cudaFree(ptr);
  g_bytes_live.fetch_sub(rounded, std::memory_order_relaxed);
  g_free_count.fetch_add(1, std::memory_order_relaxed);
  if (g_verbose) fprintf(stderr, "[univamp] free   %p size=%zu\n", ptr, rounded);
}

// Physically release all pooled blocks (call when idle / between runs, never during capture).
void univamp_empty_cache() {
  std::lock_guard<std::mutex> lk(g_mtx);
  for (auto& kv : g_free)
    for (void* p : kv.second) { cudaFree(p); g_size.erase(p); }
  g_free.clear();
}

// --- stats accessors -------------------------------------------------------------------
unsigned long long univamp_alloc_count()    { return g_alloc_count.load(); }
unsigned long long univamp_free_count()     { return g_free_count.load(); }
unsigned long long univamp_cache_hits()     { return g_cache_hits.load(); }
unsigned long long univamp_cache_miss()     { return g_cache_miss.load(); }
unsigned long long univamp_capture_miss()   { return g_capture_miss.load(); }
unsigned long long univamp_bytes_live()     { return g_bytes_live.load(); }
unsigned long long univamp_bytes_peak()     { return g_bytes_peak.load(); }
unsigned long long univamp_bytes_total()    { return g_bytes_total.load(); }
unsigned long long univamp_bytes_phys()     { return g_bytes_phys.load(); }
unsigned long long univamp_prefetch_count() { return g_prefetch_count.load(); }
void univamp_reset_stats() {
  g_alloc_count=0; g_free_count=0; g_cache_hits=0; g_cache_miss=0; g_capture_miss=0;
  g_bytes_peak=g_bytes_live.load(); g_bytes_total=0; g_bytes_phys=0; g_prefetch_count=0;
}

}  // extern "C"
