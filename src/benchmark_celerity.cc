#include <celerity/celerity.h>
#include <chrono>


const celerity::range<1> global_range = 10000;
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

template<typename Submit>
void benchmark(const char *name, Submit submit, celerity::distr_queue &q, const celerity::buffer<float> &buffer,
               const celerity::experimental::host_object<void> &obj) {
    const auto start = std::chrono::system_clock::now();
    for (int i = 0; i < samples; ++i) {
        submit(q, buffer, obj);
    }
    const auto end = std::chrono::system_clock::now();
    const auto time_per_iteration = (end - start) / samples;
    CELERITY_INFO("{}: {} us / iter", name,
                  std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(time_per_iteration).count());
}

int main() {
    celerity::distr_queue q;
    celerity::buffer<float> buffer{global_range};
    celerity::experimental::host_object obj;
    benchmark("sync only", sync_only, q, buffer, obj);
    benchmark("host tasks without dependencies", host_tasks_without_dependencies, q, buffer, obj);
    benchmark("host tasks with buffer dependencies", host_tasks_with_buffer_dependencies, q, buffer, obj);
    benchmark("host tasks with side effect dependencies", host_tasks_with_side_effect_dependencies, q, buffer, obj);
}
