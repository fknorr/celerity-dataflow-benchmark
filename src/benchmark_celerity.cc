#include <celerity/celerity.h>
#include <chrono>


const celerity::range<1> global_range = 10000;
constexpr int warmup = 100;
constexpr int samples = 1000;


[[gnu::noinline]] void sync_only(celerity::distr_queue &q, const celerity::buffer<float> &,
                                 const celerity::experimental::host_object<void> &) {
    q.slow_full_sync();
}

[[gnu::noinline]] void host_tasks_without_dependencies(celerity::distr_queue &q, const celerity::buffer<float> &,
                                                       const celerity::experimental::host_object<void> &) {
    q.submit([](celerity::handler &cgh) {
        cgh.host_task(global_range, [](celerity::partition<1>) {});
    });
    q.submit([](celerity::handler &cgh) {
        cgh.host_task(global_range, [](celerity::partition<1>) {});
    });
    q.slow_full_sync();
}

[[gnu::noinline]] void
host_tasks_with_buffer_dependencies(celerity::distr_queue &q, const celerity::buffer<float> &buffer,
                                    const celerity::experimental::host_object<void> &) {
    q.submit([=](celerity::handler &cgh) {
        celerity::accessor acc{buffer, cgh, celerity::access::one_to_one{}, celerity::write_only_host_task,
                               celerity::no_init};
        cgh.host_task(global_range, [=](celerity::partition<1> it) {
            for (size_t i = 0; i < it.get_subrange().range[0]; ++i) {
                acc[it.get_subrange().offset.get(0) + i] *= 3.14;
            }
        });
    });
    q.submit([=](celerity::handler &cgh) {
        celerity::accessor acc{buffer, cgh, celerity::access::one_to_one{}, celerity::read_write_host_task};
        cgh.host_task(global_range, [=](celerity::partition<1> it) {
            for (size_t i = 0; i < it.get_subrange().range[0]; ++i) {
                acc[it.get_subrange().offset.get(0) + i] *= 2;
            }
        });
    });
    q.slow_full_sync();
}

[[gnu::noinline]] void
host_tasks_with_side_effect_dependencies(celerity::distr_queue &q, const celerity::buffer<float> &,
                                         const celerity::experimental::host_object<void> &obj) {
    q.submit([=](celerity::handler &cgh) {
        celerity::experimental::side_effect eff{obj, cgh};
        cgh.host_task(global_range, [](celerity::partition<1> it) {});
    });
    q.submit([=](celerity::handler &cgh) {
        celerity::experimental::side_effect eff{obj, cgh};
        cgh.host_task(global_range, [](celerity::partition<1> it) {});
    });
    q.slow_full_sync();
}

int size, rank;

template<typename Submit>
void benchmark(const char *name, Submit submit, celerity::distr_queue &q, const celerity::buffer<float> &buffer,
               const celerity::experimental::host_object<void> &obj) {
    std::vector<uint64_t> sample_ns(samples);
    for (int i = 0; i < warmup; ++i) {
        submit(q, buffer, obj);
    }
    for (int i = 0; i < samples; ++i) {
        const auto start = std::chrono::system_clock::now();
        submit(q, buffer, obj);
        const auto end = std::chrono::system_clock::now();
        sample_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    std::vector<uint64_t> all_ranks_sample_ns(rank == 0 ? samples * size : 0);
    MPI_Gather(sample_ns.data(), samples, MPI_UINT64_T, all_ranks_sample_ns.data(), samples, MPI_UINT64_T, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        fmt::print("{}", name);
        for (int i = 0; i < all_ranks_sample_ns.size(); ++i) {
            fmt::print("{}{}", i % samples == 0 ? ';' : ',', all_ranks_sample_ns[i]);
        }
        fmt::print("\n");
    }
}

int main(int argc, char **argv) {
    celerity::runtime::init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        fmt::print("benchmark;rank;samples...\n");
    }

    celerity::distr_queue q;
    celerity::buffer<float> buffer{global_range};
    celerity::experimental::host_object obj;

    benchmark("sync only", sync_only, q, buffer, obj);
    benchmark("host tasks without dependencies", host_tasks_without_dependencies, q, buffer, obj);
    benchmark("host tasks with buffer dependencies", host_tasks_with_buffer_dependencies, q, buffer, obj);
    benchmark("host tasks with side effect dependencies", host_tasks_with_side_effect_dependencies, q, buffer, obj);
}
