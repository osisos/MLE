// CUDA 11.8+ examples:
//   nvcc -std=c++17 -O2 -arch=sm_80 mbarrier_sm8x.cu -o mbarrier_sm8x
//   nvcc -std=c++17 -O2 -arch=sm_86 mbarrier_sm8x.cu -o mbarrier_sm8x
//   nvcc -std=c++17 -O2 -arch=sm_89 mbarrier_sm8x.cu -o mbarrier_sm8x
// Run: ./mbarrier_sm8x
// One CTA; each thread copies 16 bytes per round. No partial tiles.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "This example requires compute capability 8.0 or newer."
#endif

template <int THREADS>
__global__ void copy_via_mbarrier(const uint4* __restrict__ input,
                                 uint4* __restrict__ output,
                                 int rounds) {
    static_assert(THREADS >= 64 && THREADS <= 1024 && THREADS % 32 == 0);
    // Launch exactly <<<1, THREADS>>>. Each uint4 occupies 16 bytes.
    __shared__ __align__(16) uint4 tile[THREADS];
    __shared__ __align__(8) unsigned long long bar;

    const unsigned tid = threadIdx.x;
    const unsigned bar_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(&bar));
    const unsigned dst_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(&tile[tid]));

    // 1. Initialize once: E=P=THREADS, phase=0. Publish to all threads.
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                     :: "r"(bar_addr), "n"(THREADS) : "memory");
    }
    __syncthreads();

    for (int round = 0; round < rounds; ++round) {
        const int base = round * THREADS;

        // 2. Issue this thread's 16-byte global -> shared copy.
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                     :: "r"(dst_addr), "l"(input + base + tid) : "memory");

        // 3. Default (no .noinc): add one pending arrival now; the
        //    completion of this thread's preceding copies arrives later.
        asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];"
                     :: "r"(bar_addr) : "memory");

        // 4. This thread arrives, P-=1, and obtains its current-phase token.
        unsigned long long token;
        asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                     : "=l"(token) : "r"(bar_addr) : "memory");

        // Independent computation could go here. Do not read tile yet.

        // 5. SM 8.x uses test_wait, not the SM 9.x try_wait instruction.
        unsigned done;
        do {
            asm volatile(
                "{\n\t"
                "  .reg .pred p;\n\t"
                "  mbarrier.test_wait.shared.b64 p, [%1], %2;\n\t"
                "  selp.u32 %0, 1, 0, p;\n\t"
                "}"
                : "=r"(done) : "r"(bar_addr), "l"(token) : "memory");
        } while (!done);

        // 6. A successful wait makes all tracked copies visible. Read
        //    the corresponding lane in the NEXT warp, not our own data.
        const unsigned peer = (tid + 32) % THREADS;
        output[base + tid] = tile[peer];

        // 7. Finish consuming this tile before any thread overwrites it.
        //    The mbarrier has already advanced phase and reset P=E.
        __syncthreads();
    }

    // 8. All users and asynchronous arrivals are finished. Retire object.
    if (tid == 0) {
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     :: "r"(bar_addr) : "memory");
    }
}

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) check_cuda((call), #call)

int main() {
    constexpr int threads = 128;
    constexpr int rounds = 3;  // Exercise phase 0 -> 1 -> 2 -> 3.
    constexpr int count = threads * rounds;
    constexpr size_t bytes = count * sizeof(uint4);

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    if (device.major < 8) {
        std::fprintf(stderr, "Requires compute capability >= 8.0; found %d.%d\n",
                     device.major, device.minor);
        return EXIT_FAILURE;
    }

    std::vector<uint4> input(count), output(count);
    for (unsigned i = 0; i < count; ++i) {
        input[i] = make_uint4(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
    }

    uint4* device_input = nullptr;
    uint4* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), bytes));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));

    copy_via_mbarrier<threads><<<1, threads>>>(device_input, device_output, rounds);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    for (int round = 0; round < rounds; ++round) {
        for (int tid = 0; tid < threads; ++tid) {
            const uint4 actual = output[round * threads + tid];
            const uint4 expected = input[round * threads + (tid + 32) % threads];
            if (actual.x != expected.x || actual.y != expected.y ||
                actual.z != expected.z || actual.w != expected.w) {
                std::fprintf(stderr, "FAIL: round=%d thread=%d\n", round, tid);
                return EXIT_FAILURE;
            }
        }
    }

    std::printf("PASS: %d rounds, %d threads, %zu bytes/round on %s (SM %d.%d)\n",
                rounds, threads, threads * sizeof(uint4), device.name,
                device.major, device.minor);
    return EXIT_SUCCESS;
}
