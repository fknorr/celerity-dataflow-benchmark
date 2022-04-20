#include <chrono>
#include <celerity/celerity.h>


constexpr int warmup = 100;
constexpr int samples = 1000;

int num_ranks, rank;

std::vector<char> send_buf;
std::vector<char> recv_buf;


[[gnu::noinline]] void nop() {
}


[[gnu::noinline]] void barrier() {
	MPI_Barrier(MPI_COMM_WORLD);
}


[[gnu::noinline]] void isend()
{
	if (rank == 0) {
		MPI_Request req;
		for (int r = 0; r < num_ranks; ++r) {
			MPI_Isend(send_buf.data(), send_buf.size(), MPI_CHAR, r, 0, MPI_COMM_WORLD, &req);
		}
	}
}


[[gnu::noinline]] void recv_barrier()
{
	MPI_Status status;
	MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
	MPI_Barrier(MPI_COMM_WORLD);
}


[[gnu::noinline]] void improbe_mrecv_barrier()
{
	int flag = 0;
	MPI_Message message;
	MPI_Status status;
	while (!flag) {
		MPI_Improbe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &message, &status);
	}
	MPI_Mrecv(recv_buf.data(), recv_buf.size(), MPI_CHAR, &message, &status);
	MPI_Barrier(MPI_COMM_WORLD);
}


void bench_barrier(std::vector<std::chrono::nanoseconds> out_ns)
{
	MPI_Barrier(MPI_COMM_WORLD);
}


void benchmark_report(const std::string &name, const std::vector<uint64_t> &sample_ns) {
	std::vector<uint64_t> all_ranks_sample_ns(rank == 0 ? samples * num_ranks : 0);
	MPI_Gather(sample_ns.data(), samples, MPI_UINT64_T, all_ranks_sample_ns.data(), samples, MPI_UINT64_T, 0,
			MPI_COMM_WORLD);
	if (rank == 0) {
		double mean = std::accumulate(all_ranks_sample_ns.begin(), all_ranks_sample_ns.end(), 0.0)
			/ all_ranks_sample_ns.size();
		fmt::print("{:55} {:10.0f} ns\n", name, mean);
		// fmt::print("{}", name);
		// for (int i = 0; i < all_ranks_sample_ns.size(); ++i) {
		// 	fmt::print("{}{}", i % samples == 0 ? ';' : ',', all_ranks_sample_ns[i]);
		// }
		// fmt::print("\n");
	}
}


void benchmark(const std::string &name, void (*operation)()) {
	std::vector<uint64_t> sample_ns(samples);
	for (int i = -warmup; i < samples; ++i) {
		std::fill(recv_buf.begin(), recv_buf.end(), '\xff');
		const auto start = std::chrono::system_clock::now();
		operation();
		const auto end = std::chrono::system_clock::now();
		if (i >= 0) {
			sample_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		}
		if (recv_buf != send_buf) {
			fmt::print(stderr, "[rank {}] recv_buf != send_buf\n", rank);
			abort();
		}
	}

	benchmark_report(name, sample_ns);
}


class spin_monitor {
	public:
		void advance(int v) {
			value = v;
		}

		void await(int v) {
			while (value < v);
		}

	private:
		std::atomic<int> value{INT_MIN};
};


class block_monitor {
	public:
		void advance(int v) {
			{
				std::lock_guard lock{mutex};
				value = v;
			}
			cond.notify_one();
		}

		void await(int v) {
			std::unique_lock lock{mutex};
			cond.wait(lock, [&]{ return value >= v; });
		}

	private:
		int value = INT_MIN;
		std::mutex mutex;
		std::condition_variable cond;
};


void benchmark(const std::string &name, void (*producer)(), void (*consumer)()) {
	std::vector<std::chrono::steady_clock::time_point> sample_starts(samples);
	std::vector<std::chrono::steady_clock::time_point> sample_ends(samples);

	spin_monitor produced, consumed;

	std::thread consumer_thread{[&]{
		for (int i = -warmup; i < samples; ++i) {
			std::fill(recv_buf.begin(), recv_buf.end(), '\xff');

			produced.await(i);
			consumer();

			if (i >= 0) {
				sample_ends[i] = std::chrono::steady_clock::now();
			}

			if (recv_buf != send_buf) {
				fmt::print(stderr, "[rank {}] recv_buf != send_buf\n", rank);
				abort();
			}
			consumed.advance(i);
		}
	}};

	for (int i = -warmup; i < samples; ++i) {
		if (i >= 0) {
			sample_starts[i] = std::chrono::steady_clock::now();
		}
		producer();
		produced.advance(i);
		consumed.await(i);
	}

	consumer_thread.join();

	std::vector<std::uint64_t> sample_ns(samples);
	for (size_t i = 0; i < samples; ++i) {
		if (sample_ends[i] < sample_starts[i]) {
			fmt::print(stderr, "[rank {}] sample_ends[i] < sample_starts[i]\n", rank);
			abort();
		}
		sample_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(sample_ends[i] - sample_starts[i]).count();
	}
	benchmark_report(name, sample_ns);
}


int main(int argc, char **argv) {
	// MPI_Init(&argc, &argv);
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	benchmark("Timer Overhead", nop);
	benchmark("Barrier", barrier);

	for (int width: {4, 16, 64, 256, 1024, 4096, 16384}) {
		send_buf.resize(width);
		recv_buf.resize(width);
		for (size_t i = 0; i < width; ++i) {
			send_buf[i] = i % 13;
		}

		benchmark(fmt::format("{:5} byte Isend, Recv, Barrier", width), [] { isend(); recv_barrier(); });
		benchmark(fmt::format("{:5} byte Isend, {{Improbe}}, Mrecv, Barrier", width), [] { isend(); improbe_mrecv_barrier(); });
		benchmark(fmt::format("{:5} byte Isend, Thread[Recv, Barrier]", width), isend, recv_barrier);
		benchmark(fmt::format("{:5} byte Isend, Thread[{{Improbe}}, Mrecv, Barrier]", width), isend, improbe_mrecv_barrier);
	}

	MPI_Finalize();
}
