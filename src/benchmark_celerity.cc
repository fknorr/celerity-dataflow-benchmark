#include <celerity/celerity.h>
#include <chrono>


const celerity::range<1> global_range = 10000;
constexpr int warmup = 100;
constexpr int samples = 1000;
constexpr int serial_chain = 10;


[[gnu::noinline]] void reference_mpi_barrier(celerity::distr_queue &, const celerity::buffer<float> &,
                                             const celerity::experimental::host_object<void> &) {
    MPI_Barrier(MPI_COMM_WORLD);
}

[[gnu::noinline]] void sync_only(celerity::distr_queue &q, const celerity::buffer<float> &,
                                 const celerity::experimental::host_object<void> &) {
    q.slow_full_sync();
}

[[gnu::noinline]] void
host_tasks_with_data_transfers(celerity::distr_queue &q, const celerity::buffer<float> &buffer,
                               const celerity::experimental::host_object<void> &) {
    q.submit([=](celerity::handler &cgh) {
        celerity::accessor acc{buffer, cgh, celerity::access::one_to_one{}, celerity::write_only_host_task,
                               celerity::no_init};
        cgh.host_task(global_range, [=](celerity::partition<1> it) {
            for (size_t i = 0; i < it.get_subrange().range[0]; ++i) {
                acc[it.get_subrange().offset.get(0) + i] *= 3;
            }
        });
    });
    constexpr auto flip_one_to_one = [](const celerity::chunk<1> ck) {
        return celerity::subrange<1>{ck.global_size[0] - ck.offset[0] - ck.range[0], ck.range[0]};
    };
    q.submit([=](celerity::handler &cgh) {
        celerity::accessor acc{buffer, cgh, flip_one_to_one, celerity::read_write_host_task};
        cgh.host_task(global_range, [=](celerity::partition<1> it) {
            for (size_t i = 0; i < it.get_subrange().range[0]; ++i) {
                acc[it.get_subrange().offset.get(0) + i] *= 2;
            }
        });
    });
    q.slow_full_sync();
}

[[gnu::noinline]] void
host_task_chain_with_side_effects(celerity::distr_queue &q, const celerity::buffer<float> &,
                                  const celerity::experimental::host_object<void> &obj) {
    for (int i = 0; i < serial_chain; ++i) {
        q.submit([=](celerity::handler &cgh) {
            celerity::experimental::side_effect eff{obj, cgh};
            cgh.host_task(global_range, [eff](celerity::partition<1> it) {});
        });
    }
    q.slow_full_sync();
}

[[gnu::noinline]] void
host_task_chain_with_syncs(celerity::distr_queue &q, const celerity::buffer<float> &,
                           const celerity::experimental::host_object<void> &) {
    for (int i = 0; i < serial_chain; ++i) {
        q.submit([=](celerity::handler &cgh) {
            cgh.host_task(global_range, [](celerity::partition<1> it) {});
        });
        q.slow_full_sync();
    }
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
        fmt::print("benchmark;samples...\n");
    }

    celerity::distr_queue q;
    celerity::buffer<float> buffer{global_range};
    celerity::experimental::host_object obj;

    benchmark("MPI_Barrier", reference_mpi_barrier, q, buffer, obj);
    benchmark("slow_full_sync", sync_only, q, buffer, obj);
    benchmark("host_task pair with one-to-one data exchange", host_tasks_with_data_transfers, q, buffer, obj);
    benchmark("host_task chain with side effects", host_task_chain_with_side_effects, q, buffer, obj);
    benchmark("host_task chain with slow_full_sync", host_task_chain_with_syncs, q, buffer, obj);
}
