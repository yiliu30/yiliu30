# From CUDA to Intel Xe: Your First Vector Add on an Intel GPU

> **Draft · Intel XPU for CUDA Developers — Part 1 of 5**  
> Tracking: [Issue #10](https://github.com/yiliu30/yiliu30/issues/10)  
> Audience: developers familiar with CUDA who want to write their first Intel GPU kernel.

If you already know how to launch a CUDA kernel, you know most of the concepts needed to get started with Intel GPUs. The names change, and so does some of the hardware organization, but the first program can be almost identical: add two arrays.

In this post, we will compute

\[
C[i] = A[i] + B[i], \qquad 0 \le i < N.
\]

We will start with familiar CUDA concepts, introduce their Intel counterparts, run a SYCL kernel, and trace one element through the hardware. We will **not** optimize matrix multiplication or use XMX yet.

**Our rule throughout this series: whenever we introduce an Intel concept, show the closest CUDA counterpart—and say when the analogy is imperfect.**

## 1. What is an Intel XPU?

A familiar NVIDIA setup has three layers:

- **Hardware:** an NVIDIA GPU, such as a Hopper or Blackwell device.
- **Programming model:** CUDA kernels, threads, blocks, and streams.
- **Toolchain:** the CUDA toolkit, including the <code>nvcc</code> compiler.

The roughly corresponding Intel setup is:

- **Hardware:** an Intel **Xe GPU**, such as an Arc Pro B60/B70 or the Xe3P-based Crescent Island GPU.
- **Programming model:** **SYCL**, a C++ model for heterogeneous programming. A SYCL **work-item** corresponds approximately to a CUDA **thread**.
- **Toolchain:** **oneAPI** and Intel's DPC++ compiler, invoked with <code>icpx -fsycl</code>, serving a role similar to <code>nvcc</code>.

**XPU** is a broad Intel term for heterogeneous compute, not the name of a single GPU instruction set. This article specifically targets **Intel Xe GPUs**. **Xe** names the GPU architecture family; **SYCL** names the programming model. They are not interchangeable.

| NVIDIA/CUDA concept | Intel Xe/SYCL counterpart | Important qualification |
|---|---|---|
| NVIDIA GPU architecture (Hopper / Blackwell) | Intel Xe GPU architecture (Xe2 / Xe3P) | Architecture families, **not** direct product or <code>sm_XX</code> equivalents |
| CUDA language and runtime | SYCL with Intel oneAPI | Similar purposes, different APIs and ecosystem |
| <code>nvcc</code> | <code>icpx -fsycl</code> | Compiler/toolchain analogy |
| CUDA kernel launch | SYCL <code>parallel_for</code> | Syntax and dispatch are different |

For our first kernel, we only need to understand how an element index is assigned to a parallel worker.

## 2. Which Intel GPUs? Xe2 and Xe3P

We will focus on two architecture generations:

- **Xe2**: the architecture of the **Intel Arc Pro B60 and B70** (Battlemage).
- **Xe3P**: the architecture of **Crescent Island (CRI)**, also discussed publicly as Intel's GPU D-series, aimed at AI inference.

Think of **Xe2** and **Xe3P** as the *type of architectural label* that **Hopper** and **Blackwell** are on NVIDIA's side. This is **not** a claim that Xe2 equals Hopper, or that Xe3P equals Blackwell in performance or supported instructions.

| Public specification | B60 | B70 | CRI |
|---|---:|---:|---:|
| **Architecture** | Xe2 | Xe2 | Xe3P |
| Xe-Cores | 20 | 32 | 32 |
| XMX engines | 160 | 256 | 256 |
| Memory | 24 GB GDDR6 | 32 GB GDDR6 | Up to 480 GB LPDDR5X |
| Memory bandwidth | 456 GB/s | 608 GB/s | Up to 1.5 TB/s (later D-series disclosure) |
| Target workload | Workstation / inference | Workstation / inference | Datacenter AI inference |
| **Highlighted low-precision formats** | FP16 / BF16 / INT8 | FP16 / BF16 / INT8 | **FP8 / MXFP8 / MXFP4** |

Sources: [B60 specifications](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html), [B70 specifications](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html), [Intel Hot Chips 2026 — CRI](https://www.intel.com/content/www/us/en/newsroom/news/client-computing/intel-outlines-architectures-for-agentic-ai-at-hot-chips-2026.html), and [Intel GPU D-series follow-up](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-accelerates-open-infrastructure-agentic-ai-era.html).

**A note on data types:** the last row highlights AI formats rather than exhaustively listing every supported scalar/instruction format. Intel has publicly emphasized FP8 and microscaled low-precision compute for Xe3P; the exact native instruction and scale-handling path for **MXFP8/MXFP4** should be checked when we implement a real matrix kernel. Supporting a format through a software library does not necessarily mean every operation executes natively.

### Xe-Core, Vector Engines, and XMX: what are the CUDA counterparts?

An Intel **Xe-Core** is loosely analogous to an NVIDIA **Streaming Multiprocessor (SM)**: both are compute building blocks in which parallel work executes. Their internal composition and scheduling are different, so do not compare raw Xe-Core counts to SM counts.

Within an Xe-Core, **Xe Vector Engines (XVEs)** handle vector and general-purpose operations. Their closest NVIDIA analogy is the CUDA FP32/INT execution pipelines (often described informally as **CUDA cores**), but there is no one-to-one engine-count mapping.

**XMX engines** accelerate matrix operations and are analogous in purpose to **NVIDIA Tensor Cores**. Our **FP32 vector addition uses the vector execution path, not XMX**.

~~~mermaid
flowchart TB
  G["Intel Xe GPU"] --> C["Xe-Core (rough analogy: NVIDIA SM)"]
  C --> V["Xe Vector Engines / XVE (rough analogy: CUDA FP32/INT execution pipelines)"]
  C --> M["XMX matrix engines (analogy: Tensor Cores)"]
  V --> A["This article: C[i] = A[i] + B[i]"]
  M --> B["Later article: matrix multiplication"]
~~~

We will study the Xe-Core internals and XMX instructions in later posts.

## 3. From CUDA to SYCL: the execution model

Here is the vocabulary you need to translate a basic CUDA kernel.

| NVIDIA CUDA | Intel SYCL | Meaning for vector add |
|---|---|---|
| CUDA thread | **Work-item** | Computes an element of the output |
| Thread block | **Work-group** | Group of work-items launched together |
| Warp (32 threads) | **Sub-group** | Cooperative SIMD-related group; supported widths depend on device/compiler |
| Grid + block dimensions | **ND-range** (global and local ranges) | Defines the total work and the size of each group |
| <code>threadIdx.x</code> | <code>item.get_local_id(0)</code> | Position within one work-group |
| <code>blockIdx.x</code> | <code>item.get_group(0)</code> | Work-group index |
| <code>blockIdx.x * blockDim.x + threadIdx.x</code> | <code>item.get_global_id(0)</code> | Global element index |
| CUDA stream | SYCL **queue** | Approximate submission/ordering counterpart; details differ |
| <code>cudaMallocManaged</code> | SYCL **USM** <code>malloc_shared</code> | Shared-access allocation; migration semantics are not identical |
| CUDA shared memory | Intel **SLM**, exposed with SYCL local memory | Work-group-shared scratchpad; **not needed here** |

The most important difference to remember: an NVIDIA warp is fixed at **32 threads**, whereas a SYCL sub-group is **not inherently warp32**. Intel GPU kernels commonly use sub-group sizes such as 8, 16, or 32, depending on architecture, compiler, and kernel constraints.

For this first example, we choose a **work-group size of 256** but let the compiler choose the sub-group width. A work-group size is *not* a request for SIMD256.

### The same 1D launch

In CUDA, a typical launch would use 256 threads per block:

~~~cpp
constexpr int BLOCK = 256;
const int grid = (N + BLOCK - 1) / BLOCK;
vector_add<<<grid, BLOCK>>>(a, b, c, N);
~~~

In SYCL, the corresponding arrangement is an <code>nd_range</code> with a **global size** rounded up to a multiple of 256 and a **local size** of 256:

~~~cpp
constexpr std::size_t LOCAL = 256;
const std::size_t global = ((N + LOCAL - 1) / LOCAL) * LOCAL;
sycl::nd_range<1> launch{
    sycl::range<1>(global), sycl::range<1>(LOCAL)};
~~~

The global size is the total **number of logical work-items**, not the number of work-groups. If <code>N = 1000</code>, we launch <code>1024</code> work-items in **four** work-groups; the last 24 work-items must do nothing.

## 4. Your first Intel GPU kernel: vector add

### CUDA: the familiar version

~~~cpp
__global__ void vector_add(
    const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
~~~

Every CUDA thread calculates its global index, loads two floats, performs one FP32 addition, and stores one float.

### Intel SYCL: the equivalent kernel

~~~cpp
q.parallel_for(
    sycl::nd_range<1>{sycl::range<1>(global),
                      sycl::range<1>(256)},
    [=](sycl::nd_item<1> item) {
      std::size_t i = item.get_global_id(0);
      if (i < N) {
        c[i] = a[i] + b[i];
      }
    }).wait_and_throw();
~~~

The operation is unchanged. The main differences are how we declare the launch and obtain the global index:

- CUDA's **kernel launch** becomes a SYCL **queue submission** with <code>parallel_for</code>.
- CUDA's **thread ID calculation** becomes <code>get_global_id(0)</code>.
- CUDA's **block size of 256** becomes the SYCL **local range of 256**.
- The bounds check remains necessary because the total launched work is rounded up.
- Calling <code>wait_and_throw()</code> completes the submitted work and propagates asynchronous failures before the CPU reads the result. This is comparable in intent—but not API semantics—to CUDA synchronization.

### A complete runnable example

The full source, including GPU selection, Unified Shared Memory (USM) allocation, initialization, synchronization, numerical validation, and cleanup, lives here:

**[examples/vector_add_sycl.cpp](examples/vector_add_sycl.cpp)**

It uses **FP32**, selects a GPU explicitly using <code>sycl::gpu_selector_v</code>, and prints **PASS** when the output is correct. The source uses <code>malloc_shared</code> to make the host-side validation easy. Unlike <code>malloc_device</code>, this allocation is accessible from both host and device (subject to device/platform support).

With the Intel oneAPI DPC++ compiler environment configured:

~~~bash
icpx -O2 -fsycl blogs/intel-xpu/examples/vector_add_sycl.cpp -o vector_add_sycl
./vector_add_sycl
~~~

A typical successful run looks like this (device name varies):

~~~text
Device: Intel(R) Arc(TM) Pro B70 Graphics
PASS
~~~

The run above is **illustrative**, not a claimed benchmark or a test performed for this draft. You need a working Intel GPU driver/runtime and a toolchain version that supports the chosen device. The program deliberately fails if no SYCL GPU is found; it does not silently fall back to the CPU.

**Why not optimize yet?** We first want a correct kernel with a transparent CUDA-to-SYCL mapping. Once it works, we can discuss device-only allocation, copies, profiling, and better vectorization.

## 5. What actually happens on the Intel GPU?

Let's follow the execution of the same kernel, first logically and then at a high hardware level.

### 5.1 Work-items map to array elements

Suppose <code>N = 1024</code> and each work-group has 256 work-items.

~~~text
CUDA                                      Intel SYCL
Grid                                      ND-range global = 1024
  Block 0 : threads   0 ... 255            Work-group 0 : items   0 ... 255
  Block 1 : threads 256 ... 511            Work-group 1 : items 256 ... 511
  Block 2 : threads 512 ... 767            Work-group 2 : items 512 ... 767
  Block 3 : threads 768 ...1023            Work-group 3 : items 768 ...1023
~~~

Each SYCL work-item calculates:

~~~text
i = get_global_id(0)
  = group_id * local_size + local_id

CUDA counterpart:
i = blockIdx.x * blockDim.x + threadIdx.x
~~~

That means work-item 579 computes **C[579]**. In our runnable example, <code>A[579] = 579</code> and <code>B[579] = 713</code>, so <code>C[579] = 1292</code>.

### 5.2 Work-groups contain sub-groups, not warps

CUDA schedules threads in **warps of 32**. The nearest Intel/SYCL concept is the **sub-group**, which is associated with the compiler's SIMD execution mapping.

For illustration, *if* the compiled kernel uses subgroup width 16:

~~~text
One work-group = 256 work-items
  Sub-group 0: local IDs   0 ... 15
  Sub-group 1: local IDs  16 ... 31
  ...
  Sub-group 15: local IDs 240 ... 255

CUDA: 256-thread block / warp32 = 8 warps
SYCL: 256-item group / subgroup16 = 16 sub-groups (illustrative)
~~~

For work-group 2, sub-group 4, lane 3, the global index would be:

~~~text
2 * 256 + 4 * 16 + 3 = 579
~~~

This is a **logical illustration**, not proof of an actual hardware lane mapping for every compiled kernel. The work-group-to-hardware scheduling, supported sub-group widths, and SIMD instruction selection depend on the selected Intel GPU and compiler.

You can inspect the reported widths on your GPU via:

~~~cpp
auto sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
~~~

This gives supported widths, not necessarily the width chosen for a particular compiled kernel.

### 5.3 Two loads, one add, one store

A work-item's logical operation is:

~~~mermaid
flowchart LR
  A["A[i] in GPU global memory"] --> L1["load"]
  B["B[i] in GPU global memory"] --> L2["load"]
  L1 --> FP["FP32 add on Xe Vector Engine"]
  L2 --> FP
  FP --> S["store"]
  S --> C["C[i] in GPU global memory"]
~~~

The same conceptual CUDA flow is **global loads → FP32 execution pipeline → global store**. On Intel Xe, we refer to the **Xe Vector Engine** for this computation; there is no Tensor Core/XMX dot product involved.

For large FP32 arrays, each element requires a *logical minimum* of 8 bytes read (two 4-byte floats) and 4 bytes written (one 4-byte float): **12 bytes per output element** before considering caching, migration, and other overheads. There is only **one floating-point addition** per element. That makes this a useful example of a typically **memory-bandwidth-bound** kernel, not a demonstration of XMX throughput.

Actual memory transactions can differ from 12 bytes due to caches, coalescing, allocation placement, and runtime behavior—especially with shared USM.

### 5.4 Putting the layers together

~~~text
NVIDIA CUDA                              Intel Xe / SYCL
---------------------------------------------------------------
CUDA grid                                SYCL ND-range
  └── thread block                        └── work-group
       └── warp32                              └── sub-group (device-dependent)
            └── thread                           └── work-item

Streaming Multiprocessor (SM)            Xe-Core (approximate)
FP32/INT execution pipeline              Xe Vector Engine (approximate)
Tensor Core                              XMX (not used for vector add)
Global memory                            Device global memory
~~~

This is the mental map we need before diving into Intel GPU internals.

## What you should remember

1. **Xe2/Xe3P ↔ Hopper/Blackwell** is a comparison of GPU *architecture generations*, not a 1:1 compute capability mapping.
2. **CUDA thread/block/warp ↔ SYCL work-item/work-group/sub-group** is a useful programming analogy—but **sub-group width is not always 32**.
3. **The vector-add arithmetic is identical** in CUDA and SYCL. Most changes are in launch syntax, memory APIs, and indexing.
4. **Vector add runs on Xe Vector Engines**, analogous in purpose to NVIDIA's general FP32 execution pipelines—not XMX/Tensor Cores.
5. **A correct, runnable kernel comes before performance tuning.**

**Next in the series:** *Inside an Xe-Core* — hardware threads, SIMD execution, Vector Engines, registers, and how they affect a vectorized kernel or reduction.

---

## References

- [Intro to AMD GPUs — From CUDA to CDNA](https://jino-rohit.github.io/blogs/19_amd_intro.html) — structure inspiration.
- [Intel Arc Pro B60 product specifications](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html).
- [Intel Arc Pro B70 product specifications](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html).
- [Intel Outlines Architectures for Agentic AI at Hot Chips 2026](https://www.intel.com/content/www/us/en/newsroom/news/client-computing/intel-outlines-architectures-for-agentic-ai-at-hot-chips-2026.html).
- [Intel GPU D-series / Crescent Island update](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-accelerates-open-infrastructure-agentic-ai-era.html).
- [Intel oneAPI vector-add example](https://www.intel.com/content/www/us/en/developer/articles/code-sample/vector-add.html).
- [Intel: Thread Mapping and GPU Occupancy](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html).
- [Intel oneAPI DPC++ compiler](https://github.com/intel/llvm).

> **Draft review TODO:** Verify the MXFP8/MXFP4 native instruction/scale-handling details on Xe3P using Intel's latest architecture/ISA documentation; run the sample on B70/B60; replace the illustrative console output with actual measured output; review any device-specific terminology.
