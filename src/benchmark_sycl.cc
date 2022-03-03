#include <chrono>
#include <sycl/sycl.hpp>

[[gnu::noinline]] void submit_and_wait(sycl::queue& q) {
	auto evt = q.submit([](sycl::handler& cgh) { cgh.single_task([] {}); });
	evt.wait();
}

constexpr int N = 1000;

int main() {
	sycl::queue q;
	const auto start = std::chrono::system_clock::now();
	for(int i = 0; i < N; ++i) {
		submit_and_wait(q);
	}
	const auto end = std::chrono::system_clock::now();
	const auto time_per_iteration = (end - start) / N;
	std::clog << std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(time_per_iteration).count() << " us / iter\n";
}
