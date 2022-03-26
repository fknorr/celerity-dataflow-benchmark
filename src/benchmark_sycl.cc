#include <celerity/celerity.h>
#include <chrono>


const sycl::range<1> global_range = 10000;
constexpr int samples = 1000;


[[gnu::noinline]] void sync_only(sycl::queue &q, sycl::buffer<float> &) {
    q.wait();
}

[[gnu::noinline]] void kernels_without_dependencies(sycl::queue &q, sycl::buffer<float> &) {
    q.submit([](sycl::handler &cgh) {
        cgh.parallel_for<class no_deps_1>(global_range, [](sycl::item<1>) {});
    });
    q.submit([](sycl::handler &cgh) {
        cgh.parallel_for<class no_deps_2>(global_range, [](sycl::item<1>) {});
    });
    q.wait();
}

[[gnu::noinline]] void kernels_with_buffer_dependencies(sycl::queue &q, sycl::buffer<float> &buffer) {
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc{buffer, cgh, sycl::write_only};
        cgh.parallel_for<class buffer_deps_1>(global_range, [=](sycl::item<1> it) { acc[it] = 3.14f; });
    });
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc{buffer, cgh, sycl::read_write};
        cgh.parallel_for<class buffer_deps_2>(global_range, [=](sycl::item<1> it) { acc[it] *= 2; });
    });
    q.wait();
}

template<typename Submit>
void benchmark(const char *name, Submit submit, sycl::queue &q, sycl::buffer<float> &buffer) {
    const auto start = std::chrono::system_clock::now();
    for (int i = 0; i < samples; ++i) {
        submit(q, buffer);
    }
    const auto end = std::chrono::system_clock::now();
    const auto time_per_iteration = (end - start) / samples;
    fmt::print(stderr, "{}: {} us / iter\n", name,
               std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(time_per_iteration).count());
}

int main() {
    // Re-use celerity's unified device selection
    celerity::runtime::init(nullptr, nullptr);
    sycl::queue &q = celerity::detail::runtime::get_instance().get_device_queue().get_sycl_queue();

    sycl::buffer<float> buffer{global_range};
    benchmark("sync only", sync_only, q, buffer);
    benchmark("device kernels without dependencies", kernels_without_dependencies, q, buffer);
    benchmark("device kernels with buffer dependencies", kernels_with_buffer_dependencies, q, buffer);
}
