# Binary Tree Search on GPU – Project Report

## 1. Introduction

Binary trees are widely used in computer science for searching, sorting, and indexing. In this project, we implement a **full binary tree search on a GPU**, compare it with a CPU baseline, and study which traversal and parallelisation strategy is most suitable for GPUs.

Each leaf of the binary tree stores a word from a large synthetic database. Given one or more keywords, the goal is to count how many leaves contain each keyword. The assignment requires:

- A full binary tree (every internal node has exactly two children).

- A GPU implementation of the search.

- Exploration of traversal mechanisms (pre-order vs post-order, breadth-first vs depth-first).

- Comparison between CPU and GPU implementations.

- At least one optimised GPU version beyond the basic implementation.

Our implementation fulfils all these requirements and includes automated testing and benchmarking.

---

## 2. Problem formulation

### 2.1 Full binary tree

A **full binary tree** is a binary tree in which every internal node has exactly two children.

For a full binary tree with `L` leaves:

- Number of internal nodes = `L − 1`.
- Total number of nodes =`2L − 1`.

In this project we focus primarily on the **leaf level**, because each leaf stores one word from the database.

### 2.2 Data model – words at leaves

- We generate a large synthetic database of **fixed‑length words** (8 uppercase characters per word).
- Each leaf of the tree stores exactly **one word**.
- We choose a **target keyword** (for example `"A"`, padded to length 8) and insert it into each leaf with probability `p ≈ 0.01`. Otherwise the leaf stores a random non‑target word.
- During generation we also count how many times we explicitly wrote the target. This is the **expected number of occurrences** and provides a quick sanity check.

Formally, given a set of leaf words `w[0..L−1]` and a keyword `k`, the task is to compute:

\[
count(k) = |{ i \mid 0 \le i < L,\ w[i] = k }|.
\]

For multiple keywords `k_0, k_1, ..., k_{K-1}` we compute `count(k_j)` for all `j`.

---

## 3. Design and methodology

### 3.1 Tree representation and full‑tree constraint

The tree is represented in **array (heap) layout**:

- If we conceptually number nodes level by level, the entire tree can be stored in a contiguous array.
- Leaves occupy a contiguous block at the end of this array.

In practice, for searching we only need the **leaf words**, so we store them in a flat array of size `leafCount × wordLen`.

The user specifies a requested number of leaves `N`. To ensure the tree is full, we compute:

- `leafCount = nextPowerOfTwo(N)` – the smallest power of two greater than or equal to `N`.
- `internalNodes = leafCount − 1`.
- `totalNodes   = internalNodes + leafCount = 2 × leafCount − 1`.

This guarantees that every internal node has exactly two children and that the resulting structure is a full binary tree.

### 3.2 Traversal and search strategy

The assignment asks to study different traversal mechanisms (pre‑order vs post‑order, breadth‑first vs depth‑first). Classical traversals such as **pre‑order** and **post‑order** are depth‑first and are typically implemented recursively or with an explicit stack.

On a GPU, recursive depth‑first search has several drawbacks:

- **Recursion and dynamic parallelism** introduce overhead and are limited in maximum depth.
- Depth‑first traversals are relatively **sequential**: they follow a path from the root to a leaf, which does not expose much parallel work.
- Access patterns tend to be **irregular**, which reduces memory coalescing on the GPU.

Instead, we choose a **breadth‑first / level‑order strategy, focused on the leaf level**:

- The leaves of a full binary tree form a single level containing `L` nodes.
- Each leaf search is **independent** of all others.
- We launch **one GPU thread per leaf** and process all leaves in parallel.

This can be viewed as a breadth‑first traversal of the last level of the tree. It is well‑suited to GPUs because:

- It avoids recursion and dynamic parallelism entirely.
- It provides massive data parallelism (up to millions of independent leaf checks).
- Leaves are stored contiguously in memory, giving **regular and coalesced memory access**.

Although we describe the approach as a breadth-first traversal, in practice the full tree structure is implicit, and the search operates directly on the contiguous leaf array representing the final level of the tree.

### 3.3 Data generation

- Use a pseudo‑random number generator (`std::mt19937`) to create random uppercase characters from `A`–`Z`.
- For each of `leafCount` leaves:
  - With probability `p`, store the padded target keyword and increment `expectedCount`.
  - Otherwise, store a random word (padded or truncated to 8 characters).
- The resulting vector of words is flattened into a contiguous `char` buffer so that the GPU can load words with simple pointer arithmetic.

### 3.4 CPU search implementation

The CPU reference implementation is straightforward:

1. Iterate through all leaf words.
2. Compare each word with the padded target keyword.
3. If they match, increment a counter.

The CPU function returns the number of matches. We measure CPU time using `std::chrono::steady_clock` and report the elapsed time in milliseconds. This gives a **baseline** for both correctness and performance.

### 3.5 Basic GPU implementation – single keyword (Option 1)

The first GPU kernel, `searchKernel`, implements the basic search for a **single keyword**:

- Grid configuration:
  - `threadsPerBlock = 256`.
  - `blocks = ceil(leafCount / threadsPerBlock)`.
- Each thread handles exactly one leaf index `tid`:

  ```text
  tid = blockIdx.x * blockDim.x + threadIdx.x
  if tid >= leafCount: return
  load word w[tid]
  compare w[tid] with target
  if equal: atomicAdd(globalCount, 1)
  ```

Key properties:

- **Traversal style:** breadth‑first, processing the entire leaf level in parallel.
- **Synchronisation:** uses `atomicAdd` to safely increment a single global counter for the number of matches.
- **Timing:** measured with CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`).

This kernel corresponds to **Option 1 – searching one keyword** in the assignment.

### 3.6 Optimised GPU implementation – multiple keywords (Option 2)

To exploit more parallelism, we implement a second kernel, `searchKernelMulti`, which searches **multiple keywords in a single pass** over the leaves.

- We maintain `K` target keywords in device memory and `K` independent counters.
- Each thread still processes one leaf word `w[tid]`, but compares it to all keywords:

  ```text
  for t in 0..K-1:
      if w[tid] == target[t]:
          atomicAdd(count[t], 1)
  ```

- In the current implementation `K = 4`.
- Keyword 0 is always the main target used in the single‑keyword search. The remaining keywords are random.

This kernel implements **Option 2 – searching multiple keywords** from the assignment. It is more “optimised” because:

- We reuse the same memory reads to evaluate several keywords.
- We perform many more comparisons per kernel launch.
- The effective throughput in **leaf‑comparisons per second** is much higher.

### 3.7 Testing and automation

To make testing and benchmarking easier, the project provides a simple Makefile:

- `make` – builds the CUDA binary `binary_tree_gpu`.
- `make test` – runs a small set of correctness tests at different tree sizes.
- `make benchmark` – runs the same configurations but used for collecting performance numbers.

For all runs we check that the program prints:

> `[CHECK] CPU and GPU results match for single and multi-keyword cases.`

indicating that CPU results equal both GPU results.

---

## 4. Experimental setup

### 4.1 Hardware and software

(You should fill in the exact details for your machine.)

- **GPU:** NVIDIA GPU (sm_60)(model and compute capability).
- **CPU:** (8-core CPU).
- **CUDA Toolkit:** version installed on the lab machine.
- **Compiler command:**

  ```bash
  nvcc -arch=sm_60 -O3 -std=c++14 -o binary_tree_gpu main.cu
  ```

### 4.2 Parameters

- **Word length:** 8 characters.
- **Target insert probability:** `p = 0.01` (approximately 1% of leaves contain the keyword).
- **Number of keywords in multi‑keyword mode:** `K = 4`.
- **Threads per block:** 256.

### 4.3 Test sizes

We evaluate several tree sizes by varying the requested number of leaves. The program automatically rounds this up to the next power of two:

- `N = 65`      → `leafCount = 128`.
- `N = 6556`    → `leafCount = 8192`.
- `N = 65566`   → `leafCount = 131072`.
- Default (`N = 1,048,576`) → `leafCount = 1,048,576`.

For each configuration we run the program several times with different target strings (e.g., `A`, `HELLO`, `HELL`).

### 4.4 Measurement methodology

For each run we record:

- CPU search count and CPU time (ms).
- GPU single‑keyword count, kernel time (ms), and throughput in **leaves per second**.
- GPU multi‑keyword counts, kernel time (ms), and throughput in **leaf‑comparisons per second**.

We focus on kernel execution time and ignore host–device transfer time, treating data transfer as a one‑time cost.

---

## 5. Results

This section summarises representative results observed during testing. Exact numbers will vary slightly between runs due to randomness.

### 5.1 Correctness

For all tested sizes and keywords the following held:

- The **CPU count** equals the **GPU single‑keyword count**.
- The **CPU count** for the main keyword (keyword 0) equals the **GPU multi‑keyword count** for keyword 0.
- The additional keywords (1–3) are random and almost never appear in the data, so both CPU and GPU consistently report **0** for them.

The program prints the final confirmation line:

> `[CHECK] CPU and GPU results match for single and multi-keyword cases.`

indicating successful validation.

### 5.2 CPU vs single‑keyword GPU

A typical set of measurements (approximate values) is:

| Leaf count | CPU time (ms) | GPU single time (ms) | GPU single throughput (leaves/s)   |
|-----------:|--------------:|---------------------:|----------------------------------: |
| 128        | ≈ 0.001       | ≈ 3.07              | ≈ 4.2 × 10⁶                         |
| 8,192      | ≈ 0.04        | ≈ 2.8–4.9           | ≈ 1.7 × 10⁶ – 2.9 × 10⁶             |
| 131,072    | ≈ 0.6         | ≈ 2.7–3.6           | ≈ 3.0 × 10⁷ – 4.8 × 10⁷             |
| 1,048,576  | ≈ 2.8         | ≈ 4.98              | ≈ 2.1 × 10⁸                         |
| 134,217,728| ≈ 377         | ≈ 4.3               | ≈ 3.1 × 10¹⁰                        |


Observations:

- The CPU implementation is very fast for this simple operation (a few comparisons and a branch per leaf).
- The basic GPU kernel achieves high throughput in leaves per second, but due to **kernel launch overhead** and **atomic operations**, the **wall‑clock time** of the basic GPU version is not always faster than the CPU for the tested sizes.
- This is an important finding: GPU acceleration is not guaranteed; it depends on both problem size and computational intensity per element.

### 5.3 Single vs multi‑keyword GPU

A comparison between the basic and optimised GPU kernels (approximate values) shows:

| Leaf count | GPU single time (ms) | GPU single throughput (leaves/s) | GPU multi time (ms) | GPU multi throughput (leaf‑comparisons/s)  |
|-----------:|---------------------:|---------------------------------:|--------------------:|-------------------------------------------:|
| 8,192      | ≈ 2.8–4.9            | ≈ 1.7 × 10⁶ –2.9 × 10⁶           | ≈ 0.018–0.023       | ≈ 1.5 × 10⁹ – 1.8 × 10⁹                    |
| 131,072    | ≈ 2.7–3.6            | ≈ 3.0 × 10⁷ – 4.8 × 10⁷          | ≈ 0.017–0.019       | ≈ 2.7 × 10¹⁰ – 3.0 × 10¹⁰                  |
| 1,048,576  | ≈ 4.98               | ≈ 2.1 × 10⁸                      | ≈ 0.058             | ≈ 6.9 × 10¹⁰ – 7.2 × 10¹⁰                  |
|134,217,728 | ≈ 4.3                | ≈ 3.1 × 10¹⁰                     | ≈ 1.93              | ≈ 2.7 × 10¹¹                               |


Observations:

- Although the GPU processes many leaves in parallel, the basic single-keyword kernel does not always outperform the CPU. This is because kernel launch overhead, atomic contention, and very low computational intensity per leaf dominate execution time for small and medium problem sizes.
- The **multi‑keyword kernel** achieves extremely high effective throughput (tens of billions of comparisons per second) by reusing the same memory accesses to test multiple keywords.
- The kernel time for multi‑keyword search grows slowly with the number of leaves, thanks to high parallelism and the fact that each thread performs only a small amount of extra computational work.
- This optimised version better demonstrates the strength of the GPU compared to the basic single‑keyword kernel.
- These large-scale measurements are included to illustrate asymptotic behaviour; exact values depend on GPU model and clock stability.


---

## 6. Discussion

### 6.1 Traversal choice: BFS vs DFS

We conceptually compared two families of traversal strategies:

- **Depth‑first search (DFS):** pre‑order or post‑order traversals typically implemented via recursion or an explicit stack.
- **Breadth‑first search (BFS):** level‑order traversal that visits nodes level by level.

On GPUs, DFS has several disadvantages:

- Recursive calls and dynamic parallelism incur overhead and are limited in depth.
- DFS explores one path at a time, which exposes limited parallelism and often leads to irregular memory access patterns.

In contrast, our **BFS‑style leaf‑parallel strategy**:

- Operates on an entire level (the leaves) in parallel.
- Maps naturally to the GPU: **one thread per leaf**.
- Provides regular, contiguous memory accesses that are friendly to the GPU memory subsystem.

Therefore, breadth‑first, array‑based leaf parallelism is a better fit for this problem and for the GPU architecture than classic recursive DFS traversals.

### 6.2 Optimisation techniques

The project implements several optimisation ideas:

1. **Array / heap layout and full tree constraint**  
   The full binary tree is stored implicitly in arrays, and leaves are contiguous in memory. This maximises memory coalescing when GPU threads read leaf words.

2. **BFS leaf‑parallel traversal (no recursion)**  
   We avoid recursion and dynamic parallelism and instead process all leaves simultaneously with one thread per leaf. This exposes maximum data parallelism.

3. **Multi‑keyword search in a single pass**  
   The optimised GPU kernel tests multiple keywords in one pass, greatly increasing the amount of useful work per memory access and per kernel launch.

4. **Automated testing and benchmarking**  
   `make test` and `make benchmark` provide a repeatable way to verify correctness and gather timing and throughput statistics.

### 6.3 Limitations and future work

- **Atomic contention:**  
  Multiple threads may update the same global counter concurrently using `atomicAdd`, which can limit scalability. Future versions could use per‑block partial reductions in shared memory followed by a final global reduction.

- **More complex queries:**  
  The current implementation counts exact matches for fixed‑length words. Extensions could include prefix searches or variable‑length strings.

- **Explicit DFS implementation:**  
  Implementing a DFS version using an explicit stack in global or shared memory would allow a direct experimental comparison between DFS and BFS on the same hardware.

---

## 7. Conclusion

In this project we implemented and evaluated a **GPU‑based binary tree search** for a large synthetic word database.

- We enforced the **full binary tree** property by rounding the requested leaf count up to the next power of two.
- We stored all leaves in a contiguous array and used a **breadth‑first, level‑order, leaf‑parallel traversal** to fully exploit GPU parallelism without recursion.
- We implemented:
  - A **CPU reference search**.
  - A **basic GPU kernel** for a single keyword.
  - An **optimised GPU kernel** that searches multiple keywords in parallel.
- We validated correctness by checking that CPU and GPU results match for all configurations.
- Our experiments showed that while the basic GPU version does not always outperform the CPU in wall‑clock time for this simple operation, the optimised multi‑keyword kernel achieves extremely high comparison throughput and demonstrates the benefits of GPU parallelism.

Overall, the project meets the assignment requirements and provides a clear case study of how data layout, traversal strategy and kernel design influence performance on GPUs.
