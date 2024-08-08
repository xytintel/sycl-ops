#include <sycl/sycl.hpp>
#include <iostream>

#define SYCL_KER_STRING(var, str) \
  static const __attribute__((opencl_constant)) char var[] = str;
#define SYCL_KER_PRINTF sycl::ext::oneapi::experimental::printf
#define SYCL_K_PRINT(fmt_str, ...)           \
  {                                          \
    SYCL_KER_STRING(fmt_var, fmt_str);       \
    SYCL_KER_PRINTF(fmt_var, ##__VA_ARGS__); \
  }

template <typename in_dtype, typename out_dtype>
struct LaunchKernel {
  void operator()(sycl::nd_item<1> item) const {
    out_[0] = in_[0];
  }
  LaunchKernel(in_dtype* in, out_dtype* out) : in_(in), out_(out) {}

 private:
  in_dtype* in_;
  out_dtype* out_;
};

int main(int argc, char* argv[]) {
  sycl::queue q(
      sycl::gpu_selector_v,
      {sycl::property::queue::enable_profiling(),
       sycl::property::queue::in_order()});

  using in_dtype = float;
  using out_dtype = int64_t;

  in_dtype* in = sycl::malloc_shared<in_dtype>(1, q);
  out_dtype* out = sycl::malloc_shared<out_dtype>(1, q);

  in_dtype in_cpu[1] = {-3.4028e+38};
  q.memcpy(in, in_cpu, 1 * sizeof(in_dtype)).wait();

  std::cout << "in: " << in[0] << "\n";

  auto global_range = sycl::range<1>(1);
  auto local_range = sycl::range<1>(1);

  auto e = q.submit([&](sycl::handler& cgh) {
    auto kernel = LaunchKernel<in_dtype, out_dtype>(in, out);
    cgh.parallel_for<>(sycl::nd_range<1>(global_range, local_range), kernel);
  });
  e.wait();

  std::cout << "out: " << out[0] << "\n";

  free(in, q);
  free(out, q);
  return 0;
}
