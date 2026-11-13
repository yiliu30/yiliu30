// Part 1: Intro to Intel Xe — the same vector add as CUDA, in SYCL.
#include <sycl/sycl.hpp>

#include <cstddef>
#include <iostream>
#include <new>
#include <stdexcept>

int main() {
  try {
    constexpr std::size_t N = 1u << 20;
    constexpr std::size_t LOCAL_SIZE = 256;

    // CUDA analogy: select a GPU and create a CUDA stream.
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << '\n';

    // CUDA analogy: cudaMallocManaged(), not cudaMalloc().
    float* a = sycl::malloc_shared<float>(N, q);
    float* b = sycl::malloc_shared<float>(N, q);
    float* c = sycl::malloc_shared<float>(N, q);

    if (!a || !b || !c) {
      if (a) sycl::free(a, q);
      if (b) sycl::free(b, q);
      if (c) sycl::free(c, q);
      throw std::bad_alloc{};
    }

    for (std::size_t i = 0; i < N; ++i) {
      a[i] = static_cast<float>(i % 1024);
      b[i] = static_cast<float>((i * 3) % 1024);
      c[i] = -1.0f;
    }

    const std::size_t global_size =
        ((N + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;

    // CUDA analogy: kernel<<<ceil(N / 256), 256>>>(a, b, c, N).
    q.parallel_for(
         sycl::nd_range<1>{sycl::range<1>(global_size),
                           sycl::range<1>(LOCAL_SIZE)},
         [=](sycl::nd_item<1> item) {
           const std::size_t i = item.get_global_id(0);
           if (i < N) {
             c[i] = a[i] + b[i];
           }
         })
        .wait_and_throw();  // CUDA analogy: synchronize before reading output.

    bool passed = true;
    for (std::size_t i = 0; i < N; ++i) {
      if (c[i] != a[i] + b[i]) {
        std::cerr << "Mismatch at " << i << ": " << c[i] << '\n';
        passed = false;
        break;
      }
    }

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);

    std::cout << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
