# Binary Tree Search on GPU 

This project implements a **full binary tree search** over a large synthetic word database using **CUDA**.

- The tree is *full*: number of leaves is rounded up to the next power of two, and internal node count = leaves - 1.
- Each leaf holds a fixed-length word.
- A basic CUDA kernel performs a **breadth-first / level-order style** traversal using an array representation of the full binary tree, assigning one thread per leaf, and counts how many times a given target word appears in the leaf set (Option 1: single keyword).
- An extended CUDA kernel searches **multiple keywords in parallel** (Option 2: multiple keywords), with each thread comparing its leaf word against several target words.
- A CPU reference search is provided for correctness and performance comparison against both GPU versions.

## Key Concepts Used

- **Full binary tree**: a binary tree where every internal node has exactly two children.
- **Array (heap) representation**: tree structure is implicit; leaves are stored contiguously.
- **Breadth-first (level-order) traversal**: all leaves are processed in parallel.
- **GPU data parallelism**: one CUDA thread processes one leaf.
- **Atomic operations**: used to safely accumulate match counts across threads.

## Limitations and Notes

- Atomic updates to global counters may limit scalability for extremely large trees.
- Only exact string matches are supported.
- Internal tree nodes are implicit and not explicitly traversed.
- GPU kernel launch overhead dominates for small problem sizes.


## Files

- **main.cu**

  Contains:

  - Synthetic database generation (random uppercase words).
  - Construction of a full binary tree in **array form** (heap layout) by rounding the requested number of leaves up to the next power of two.
  - CPU search function.
  - CUDA kernel `searchKernel` (basic GPU version, single keyword) that:
    - Treats the array of leaves as the last level of a full binary tree.
    - Launches one thread per leaf, which is equivalent to parallel breadth-first traversal at the leaf level.
    - Uses an atomic counter to accumulate matches.
  - CUDA kernel `searchKernelMulti` (optimized/more parallel GPU version, multiple keywords) that:
    - Allows several target keywords to be searched in a single kernel launch.
    - Accumulates one counter per keyword.
  - Timing and throughput measurement for:
    - CPU search.
    - Single-keyword GPU search.
    - Multi-keyword GPU search.

- **Makefile**

  Simple build script using `nvcc`.

## Build

You need NVIDIA CUDA toolkit and a compatible GPU.

```bash
cd binary_tree_gpu
make
```

Set your GPU architecture if necessary (for example, `sm_70`):

```bash
make ARCH=-arch=sm_70
```

### Tests and benchmarks

You can run a small predefined **test/benchmark suite** via the Makefile:

```bash
make test       # correctness-focused runs
make benchmark  # performance-focused runs
```

Both targets currently execute three configurations:

- 65,536 leaves (2^16), keyword `A`
- 262,144 leaves (2^18), keyword `A`
- 1,048,576 leaves (2^20), keyword `A`

For each run, check that the program reports:

- `CPU and GPU results match for single and multi-keyword cases.`

For your report/presentation, record from these runs:

- CPU search time.
- GPU single-keyword time and leaves/s throughput.
- GPU multi-keyword time and leaf-comparisons/s throughput.

## Run

Usage:

```bash
./binary_tree_gpu [leaf_count] [target_word]
```

- **leaf_count** (optional): desired number of leaves in the tree.
  - The program will round this up to the next power of two to enforce a full binary tree.
  - Default: `1<<20` (1,048,576 leaves).
- **target_word** (optional): the word to search for.
  - Default: `"A"` (padded/truncated to fixed internal length).

Examples:

```bash
# Default settings
./binary_tree_gpu

# 262,144 leaves, searching for word "HELLO"
./binary_tree_gpu 262144 HELLO
```

The program prints:

- Number of leaves actually used (power-of-two).
- Total nodes in full binary tree.
- Expected occurrences of the main target generated on the host.
- CPU search count and time.
- GPU (single-keyword) search count, kernel execution time, and leaf/s throughput.
- GPU (multi-keyword) search counts for several keywords, kernel execution time, and leaf-comparisons/s throughput.

## Traversal Discussion

- The full binary tree is stored in **array (heap) layout**.
- Leaves are contiguous in memory at the last level.
- We implement a **breadth-first style search** by assigning one GPU thread per leaf and processing them all in parallel.
- This avoids recursion (which is costly and limited on GPUs) and leverages data-parallel SIMD-style execution.

This maps directly to the assignment options:

- **Option 1 – Searching one keyword**: run the program and focus on the single-keyword GPU results vs CPU. You can discuss how one thread per leaf corresponds to many parallel searches of the same keyword.
- **Option 2 – Searching multiple keywords**: use the printed multi-keyword results to show how many keywords are searched in parallel and how throughput increases.

For your report/presentation you can:

- Compare **CPU vs basic GPU vs multi-keyword GPU** in execution time and throughput (operations per second).
- Discuss why breadth-first / level-order traversal with array layout is suitable for GPUs, and why recursive depth-first traversal with dynamic parallelism is less efficient and more complex.

You can extend this framework to:

- Experiment with different traversal orders (e.g., emulating depth-first with an explicit stack in global memory).
- Vary tree size and word distributions.
- Add more detailed performance statistics (throughput, bandwidth, etc.).
