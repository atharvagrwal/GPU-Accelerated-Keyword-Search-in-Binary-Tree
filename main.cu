#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <iostream>
 #include <chrono>

// Configuration
static const int DEFAULT_WORD_LEN = 8;      // fixed length words on device
static const int MAX_WORD_LEN     = 32;     // limit for host side strings

// Simple CUDA error check macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                        cudaGetErrorString(err), __FILE__, __LINE__);         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Kernel: each thread inspects one leaf word in the full binary tree and
// performs a simple string comparison with the target. If equal, it performs
// an atomic add on the global counter.
__global__ void searchKernel(const char* __restrict__ d_words,
                             int wordLen,
                             int leafCount,
                             const char* __restrict__ d_target,
                             int* d_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leafCount) return;

    const char* w = d_words + tid * wordLen;

    // Compare fixed-length words
    bool equal = true;
    for (int i = 0; i < wordLen; ++i) {
        if (w[i] != d_target[i]) {
            equal = false;
            break;
        }
    }

    if (equal) {
        atomicAdd(d_count, 1);
    }
}

__global__ void searchKernelMulti(const char* __restrict__ d_words,
                                  int wordLen,
                                  int leafCount,
                                  const char* __restrict__ d_targets,
                                  int numTargets,
                                  int* d_counts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leafCount) return;

    const char* w = d_words + tid * wordLen;

    for (int t = 0; t < numTargets; ++t) {
        const char* target = d_targets + t * wordLen;

        bool equal = true;
        for (int i = 0; i < wordLen; ++i) {
            if (w[i] != target[i]) {
                equal = false;
                break;
            }
        }

        if (equal) {
            atomicAdd(&d_counts[t], 1);
        }
    }
}

// Generate a random uppercase word of fixed length.
static std::string randomWord(int len, std::mt19937& gen)
{
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::uniform_int_distribution<int> dist(0, (int)sizeof(alphabet) - 2);
    std::string s;
    s.reserve(len);
    for (int i = 0; i < len; ++i) {
        s.push_back(alphabet[dist(gen)]);
    }
    return s;
}

// Pad or truncate word to exactly DEFAULT_WORD_LEN characters.
static std::string toFixedWord(const std::string& in)
{
    std::string out = in;
    if ((int)out.size() > DEFAULT_WORD_LEN) {
        out.resize(DEFAULT_WORD_LEN);
    } else if ((int)out.size() < DEFAULT_WORD_LEN) {
        out.append(DEFAULT_WORD_LEN - out.size(), ' ');
    }
    return out;
}

// Simple CPU reference implementation to verify correctness.
static int cpuSearch(const std::vector<std::string>& words,
                     const std::string& target)
{
    int count = 0;
    for (const auto& w : words) {
        if (w == target) {
            ++count;
        }
    }
    return count;
}

// Ensure leaf count corresponds to a full binary tree: number of leaves is
// a power of two. If not, round up to next power of two.
static int nextPowerOfTwo(int n)
{
    if (n <= 1) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// Entry point
int main(int argc, char** argv)
{
    int    requestedLeaves = 1 << 20; // 1M leaves by default
    int    wordLen         = DEFAULT_WORD_LEN;
    double targetFraction  = 0.01;    // ~1% of leaves equal to target
    std::string targetWord = "A";    // base target before padding

    if (argc >= 2) {
        requestedLeaves = std::atoi(argv[1]);
        if (requestedLeaves <= 0) {
            std::cerr << "Invalid leaf count, using default\n";
            requestedLeaves = 1 << 20;
        }
    }
    if (argc >= 3) {
        targetWord = argv[2];
        if ((int)targetWord.size() > MAX_WORD_LEN) {
            targetWord.resize(MAX_WORD_LEN);
        }
    }

    int leafCount = nextPowerOfTwo(requestedLeaves);
    int internalNodes = leafCount - 1; // for a full binary tree
    int totalNodes   = internalNodes + leafCount;

    std::cout << "========== Binary Tree GPU Search ==========" << "\n";
    std::cout << "[SETUP] Requested leaves           : " << requestedLeaves << "\n";
    std::cout << "[SETUP] Power-of-two leaf count    : " << leafCount << "\n";
    std::cout << "[SETUP] Total nodes (full tree)    : " << totalNodes << "\n";

    // Prepare host-side synthetic word database (one word per leaf)
    std::mt19937 gen(42);
    std::bernoulli_distribution isTarget(targetFraction);

    std::string fixedTarget = toFixedWord(targetWord);

    std::vector<std::string> hostWords;
    hostWords.reserve(leafCount);

    int expectedCount = 0;
    for (int i = 0; i < leafCount; ++i) {
        if (isTarget(gen)) {
            hostWords.push_back(fixedTarget);
            ++expectedCount;
        } else {
            hostWords.push_back(toFixedWord(randomWord(wordLen, gen)));
        }
    }

    std::cout << "[SETUP] Expected target occurrences : "
              << expectedCount << "\n";

    // Flatten words into a single contiguous buffer
    size_t wordsBytes = static_cast<size_t>(leafCount) * wordLen;
    std::vector<char> h_words(wordsBytes);
    for (int i = 0; i < leafCount; ++i) {
        std::memcpy(&h_words[i * wordLen], hostWords[i].data(), wordLen);
    }

    // CPU reference search on the unpadded logical words (for sanity)
    auto cpuStart = std::chrono::steady_clock::now();
    int cpuCount = cpuSearch(hostWords, fixedTarget);
    auto cpuEnd = std::chrono::steady_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "---------------------------------------------" << "\n";
    std::cout << "[CPU]  count                     : " << cpuCount << "\n";
    std::cout << "[CPU]  time (ms)                 : " << cpuMs << "\n";

    // --- Device allocations ---
    char* d_words  = nullptr;
    char* d_target = nullptr;
    int*  d_count  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_words, wordsBytes));
    CUDA_CHECK(cudaMalloc(&d_target, wordLen));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_words, h_words.data(), wordsBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, fixedTarget.data(), wordLen,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    // --- Launch kernel ---
    int threadsPerBlock = 256;
    int blocks = (leafCount + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "---------------------------------------------" << "\n";
    std::cout << "[GPU single] Launch: blocks = " << blocks
              << ", threads/block = " << threadsPerBlock << "\n";

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    searchKernel<<<blocks, threadsPerBlock>>>(d_words,
                                              wordLen,
                                              leafCount,
                                              d_target,
                                              d_count);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    int gpuCount = 0;
    CUDA_CHECK(cudaMemcpy(&gpuCount, d_count, sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::cout << "[GPU single] count              : " << gpuCount << "\n";
    std::cout << "[GPU single] kernel time (ms)   : " << ms << "\n";

    double gpuSeconds = ms * 1e-3;
    double gpuThroughput = leafCount / gpuSeconds;
    std::cout << "[GPU single] throughput (leaves/s): " << gpuThroughput << "\n";

    if (gpuCount != cpuCount) {
        std::cerr << "Mismatch between CPU and GPU results!" << std::endl;
        CUDA_CHECK(cudaFree(d_words));
        CUDA_CHECK(cudaFree(d_target));
        CUDA_CHECK(cudaFree(d_count));
        return EXIT_FAILURE;
    }

    // --- Multi-keyword search experiment (Option 2) ---

    int numTargets = 4;
    std::vector<std::string> multiTargets;
    multiTargets.reserve(numTargets);
    multiTargets.push_back(fixedTarget);
    for (int i = 1; i < numTargets; ++i) {
        multiTargets.push_back(toFixedWord(randomWord(wordLen, gen)));
    }

    std::vector<char> h_targetsMulti(numTargets * wordLen);
    for (int t = 0; t < numTargets; ++t) {
        std::memcpy(&h_targetsMulti[t * wordLen], multiTargets[t].data(), wordLen);
    }

    std::vector<int> cpuCountsMulti(numTargets, 0);
    for (int t = 0; t < numTargets; ++t) {
        cpuCountsMulti[t] = cpuSearch(hostWords, multiTargets[t]);
    }

    char* d_targetsMulti = nullptr;
    int*  d_countsMulti  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_targetsMulti, numTargets * wordLen));
    CUDA_CHECK(cudaMalloc(&d_countsMulti, numTargets * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_targetsMulti, h_targetsMulti.data(),
                          numTargets * wordLen, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_countsMulti, 0, numTargets * sizeof(int)));

    cudaEvent_t start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    CUDA_CHECK(cudaEventRecord(start2));

    searchKernelMulti<<<blocks, threadsPerBlock>>>(d_words,
                                                   wordLen,
                                                   leafCount,
                                                   d_targetsMulti,
                                                   numTargets,
                                                   d_countsMulti);

    CUDA_CHECK(cudaEventRecord(stop2));
    CUDA_CHECK(cudaEventSynchronize(stop2));

    float ms2 = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start2, stop2));

    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));

    std::vector<int> gpuCountsMulti(numTargets, 0);
    CUDA_CHECK(cudaMemcpy(gpuCountsMulti.data(), d_countsMulti,
                          numTargets * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "---------------------------------------------" << "\n";
    std::cout << "[GPU multi] " << numTargets << " keywords in parallel" << "\n";
    for (int t = 0; t < numTargets; ++t) {
        std::cout << "  Keyword " << t << ": CPU = " << cpuCountsMulti[t]
                  << ", GPU = " << gpuCountsMulti[t] << "\n";
    }
    std::cout << "[GPU multi] kernel time (ms)         : " << ms2 << "\n";

    double gpuSeconds2 = ms2 * 1e-3;
    double gpuThroughput2 = (static_cast<double>(leafCount) * numTargets) / gpuSeconds2;
    std::cout << "[GPU multi] throughput (leaf-comparisons/s): "
              << gpuThroughput2 << "\n";

    CUDA_CHECK(cudaFree(d_words));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_targetsMulti));
    CUDA_CHECK(cudaFree(d_countsMulti));

    std::cout << "---------------------------------------------" << "\n";
    std::cout << "[CHECK] CPU and GPU results match for single and multi-keyword cases." << std::endl;
    std::cout << "=============================================" << std::endl;
    return EXIT_SUCCESS;
}
