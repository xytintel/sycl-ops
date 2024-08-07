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

template <class scalar_t, class CombineFunc>
inline scalar_t group_x_reduce(
    sycl::nd_item<2> item,
    sycl::local_ptr<scalar_t> shared,
    scalar_t value,
    CombineFunc combine) {
  int l_x = item.get_local_id(1); // x for x in range(32)
  int l_y = item.get_local_id(0); // x for x in range(4)
  int g_x = item.get_local_range(1); // 32
  int dim_x = g_x;
  auto sg = item.get_sub_group();
  int sg_size = sg.get_local_range()[0];

  if (dim_x > sg_size) {
    int base = l_x + l_y * g_x;
    shared[base] = value;
    for (int offset = dim_x / 2; offset >= sg_size; offset >>= 1) {
      item.barrier(sycl::access::fence_space::local_space);
      if (l_x < offset && l_x + offset < g_x) {
        scalar_t other = shared[base + offset];
        value = combine(value, other);
        shared[base] = value;
      }
    }
    dim_x = sg_size;
  }

  // sub-group reduction
  for (int offset = 1; offset < dim_x; offset <<= 1) {
    scalar_t other = sycl::shift_group_left(sg, value, offset);
    value = combine(value, other);
  }
  return value;
}

template <typename scalar_t>
struct ReduceFn {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a < b ? a : b;
  }
};

template <typename scalar_t>
struct LaunchKernel {
  [[intel::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<2> item) const {
    int l_x = item.get_local_id(1); // x for x in range(32)
    int l_y = item.get_local_id(0); // x for x in range(4)
    // int g_x = item.get_local_range(1); // 32
    int seg_id = l_y;
    int problem_id = l_x;
    scalar_t value;
    if (problem_id < problem_size_) {
      value = in_[seg_id * problem_size_ + problem_id];
    } else {
      value = 1;
    }
    auto compare_fn = ReduceFn<scalar_t>();
    value = group_x_reduce<scalar_t>(item, shared_, value, compare_fn);
    if (problem_id == 0)
      out_[seg_id] = value;
  }
  LaunchKernel(
      scalar_t* in,
      scalar_t* out,
      int problem_size,
      sycl::local_accessor<scalar_t> shared)
      : in_(in), out_(out), problem_size_(problem_size), shared_(shared) {}

 private:
  scalar_t* in_;
  scalar_t* out_;
  int problem_size_;
  sycl::local_accessor<scalar_t> shared_;
};

int main(int argc, char* argv[]) {
  sycl::queue q(
      sycl::gpu_selector_v,
      {sycl::property::queue::enable_profiling(),
       sycl::property::queue::in_order()});

  int nsegments = 4;
  int problem_size = 5;

  bool input_cpu[4 * 5] = {
      false, true, true, false, false, true, false, true,  false, true,
      true,  true, true, true,  true,  true, true,  false, false, false,
  };
  bool* in = sycl::malloc_shared<bool>(nsegments * problem_size, q);
  bool* out = sycl::malloc_shared<bool>(nsegments, q);
  q.memcpy(in, input_cpu, nsegments * problem_size).wait();

  std::cout << "input:\n";
  for (int i = 0; i < nsegments; i++) {
    for (int j = 0; j < problem_size; j++) {
      std::cout << in[i * problem_size + j];
    }
    std::cout << "\n";
  }

  auto global_range = sycl::range<2>(4, 32);
  auto local_range = sycl::range<2>(4, 32);

  auto e = q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<bool> local_acc(sycl::range<1>(256), cgh);
    auto kernel = LaunchKernel<bool>(in, out, problem_size, local_acc);
    cgh.parallel_for<>(sycl::nd_range<2>(global_range, local_range), kernel);
  });
  e.wait();

  std::cout << "\noutput:\n";
  for (int i = 0; i < nsegments; i++) {
    std::cout << out[i];
  }
  std::cout << "\n";

  free(in, q);
  free(out, q);
  return 0;
}
