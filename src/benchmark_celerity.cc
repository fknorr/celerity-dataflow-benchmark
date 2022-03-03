#include <celerity/celerity.h>
#include <chrono>

[[gnu::noinline]] void submit_and_wait(celerity::distr_queue& q) {
	q.slow_full_sync();
}

constexpr int N = 1000;

int main() {
	celerity::distr_queue q;
	const auto start = std::chrono::system_clock::now();
	for(int i = 0; i < N; ++i) {
		submit_and_wait(q);
	}
	const auto end = std::chrono::system_clock::now();
	const auto time_per_iteration = (end - start) / N;
	std::cout << std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(time_per_iteration).count() << "us / iter\n";
}