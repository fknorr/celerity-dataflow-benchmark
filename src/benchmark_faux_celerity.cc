#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <numeric>
#include <queue>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <mpi.h>

//////////////////////////////////////////////////////////////////
/////////////////////// FAUX CELERITY ////////////////////////////
//////////////////////////////////////////////////////////////////

namespace celerity {
namespace detail {

	template <typename T, typename UniqueName>
	class PhantomType {
	  public:
		using underlying_t = T;

		constexpr PhantomType() = default;
		constexpr PhantomType(T const& value) : value(value) {}
		constexpr PhantomType(T&& value) : value(std::move(value)) {}

		// Allow implicit conversion to underlying type, otherwise it becomes too annoying to use.
		// Luckily compilers won't do more than one user-defined conversion, so something like
		// PhantomType1<T> -> T -> PhantomType2<T>, can't happen. Therefore we still retain
		// strong typesafety between phantom types with the same underlying type.
		constexpr operator T&() { return value; }
		constexpr operator const T&() const { return value; }

	  private:
		T value;
	};

} // namespace detail
} // namespace celerity

#define MAKE_PHANTOM_TYPE(TypeName, UnderlyingT)                                                                                                               \
	namespace celerity {                                                                                                                                       \
		namespace detail {                                                                                                                                     \
			using TypeName = PhantomType<UnderlyingT, class TypeName##_PhantomType>;                                                                           \
		}                                                                                                                                                      \
	}                                                                                                                                                          \
	namespace std {                                                                                                                                            \
		template <>                                                                                                                                            \
		struct hash<celerity::detail::TypeName> {                                                                                                              \
			std::size_t operator()(const celerity::detail::TypeName& t) const noexcept { return std::hash<UnderlyingT>{}(static_cast<UnderlyingT>(t)); }       \
		};                                                                                                                                                     \
	}

MAKE_PHANTOM_TYPE(task_id, size_t)
MAKE_PHANTOM_TYPE(buffer_id, size_t)
MAKE_PHANTOM_TYPE(node_id, size_t)
MAKE_PHANTOM_TYPE(command_id, size_t)
MAKE_PHANTOM_TYPE(collective_group_id, size_t)
MAKE_PHANTOM_TYPE(reduction_id, size_t)
MAKE_PHANTOM_TYPE(host_object_id, size_t)

namespace celerity {
namespace detail {

	namespace mpi_support {

		constexpr int TAG_CMD = 0;
		constexpr int TAG_DATA_TRANSFER = 1;
		constexpr int TAG_TELEMETRY = 2;

		class single_use_data_type {
		  public:
			single_use_data_type() = default;
			single_use_data_type(MPI_Datatype dt) : dt(dt){};

			single_use_data_type(single_use_data_type&& other) noexcept { *this = std::move(other); }
			single_use_data_type& operator=(single_use_data_type&& other) noexcept {
				if(this != &other) {
					dt = other.dt;
					other.dt = MPI_DATATYPE_NULL;
				}
				return *this;
			}

			single_use_data_type(const single_use_data_type& other) = delete;
			single_use_data_type& operator=(const single_use_data_type& other) = delete;

			MPI_Datatype operator*() const { return dt; }

			~single_use_data_type() {
				if(dt != MPI_DATATYPE_NULL) { MPI_Type_free(&dt); }
			}

		  private:
			MPI_Datatype dt = MPI_DATATYPE_NULL;
		};

		single_use_data_type build_single_use_composite_type(const std::vector<std::pair<size_t, void*>>& blocks) {
			std::vector<int> block_lengths;
			block_lengths.reserve(blocks.size());
			std::vector<MPI_Aint> disps;
			disps.reserve(blocks.size());
			for(auto& b : blocks) {
				block_lengths.push_back(static_cast<int>(b.first));
				disps.push_back(reinterpret_cast<MPI_Aint>(b.second));
			}
			std::vector<MPI_Datatype> block_types(blocks.size(), MPI_BYTE);
			MPI_Datatype data_type;
			MPI_Type_create_struct(static_cast<int>(blocks.size()), block_lengths.data(), disps.data(), block_types.data(), &data_type);
			MPI_Type_commit(&data_type);
			return data_type;
		}

	} // namespace mpi_support

	enum class command_type { NOP, HORIZON, EXECUTION, PUSH, AWAIT_PUSH, REDUCTION, SHUTDOWN, SYNC };

	struct nop_data {};
	struct horizon_data {};
	struct execution_data {};
	struct push_data {};
	struct await_push_data {};
	struct reduction_data {};
	struct shutdown_data {};
	struct sync_data {
		uint64_t sync_id;
	};

	using command_data = std::variant<nop_data, horizon_data, execution_data, push_data, await_push_data, reduction_data, shutdown_data, sync_data>;

	struct command_pkg {
		command_id cid;
		command_type cmd;
		command_data data;

		command_pkg() = default;
		command_pkg(command_id cid, command_type cmd, command_data data) : cid(cid), cmd(cmd), data(data) {}
	};

	class executor {
	  public:
		executor(node_id local_nid);

		void startup();
		void shutdown();
		uint64_t get_highest_executed_sync_id() const noexcept;

	  private:
		node_id local_nid;
		std::thread exec_thrd;
		std::atomic<uint64_t> highest_executed_sync_id = {0};
		void run();
	};

	executor::executor(node_id local_nid) : local_nid(local_nid) {}

	void executor::startup() { exec_thrd = std::thread(&executor::run, this); }

	void executor::shutdown() {
		if(exec_thrd.joinable()) { exec_thrd.join(); }
	}

	uint64_t executor::get_highest_executed_sync_id() const noexcept { return highest_executed_sync_id; }

	void executor::run() {
		bool done = false;
		constexpr uint64_t NOT_SYNCING = std::numeric_limits<uint64_t>::max();
		uint64_t syncing_on_id = NOT_SYNCING;

		struct command_info {
			command_pkg pkg;
			std::vector<command_id> dependencies;
		};
		std::queue<command_info> command_queue;

		while(!done) {
			if(syncing_on_id != NOT_SYNCING) {
				MPI_Barrier(MPI_COMM_WORLD);
				highest_executed_sync_id = syncing_on_id;
				syncing_on_id = NOT_SYNCING;
			}

			MPI_Status status;
			int flag;
			MPI_Message msg;
			MPI_Improbe(MPI_ANY_SOURCE, mpi_support::TAG_CMD, MPI_COMM_WORLD, &flag, &msg, &status);
			if(flag == 1) {
				// Commands should be small enough to block here (TODO: Re-evaluate this now that we also transfer dependencies)
				command_queue.emplace<command_info>({});
				auto& pkg = command_queue.back().pkg;
				auto& dependencies = command_queue.back().dependencies;
				int count;
				MPI_Get_count(&status, MPI_CHAR, &count);
				const size_t deps_size = count - sizeof(command_pkg);
				dependencies.resize(deps_size / sizeof(command_id));
				const auto data_type = mpi_support::build_single_use_composite_type({{sizeof(command_pkg), &pkg}, {deps_size, dependencies.data()}});
				MPI_Mrecv(MPI_BOTTOM, 1, *data_type, &msg, &status);
			}

			if(syncing_on_id == NOT_SYNCING && !command_queue.empty()) {
				const auto info = command_queue.front();
				if(info.pkg.cmd == command_type::SHUTDOWN) {
					assert(command_queue.size() == 1);
					done = true;
				} else if(info.pkg.cmd == command_type::SYNC) {
					syncing_on_id = std::get<sync_data>(info.pkg.data).sync_id;
				} else {
					throw std::runtime_error("NOPE!");
				}
				command_queue.pop();
			}
		}
	}

	class runtime {
	  public:
		static void init(int* argc, char** argv[]);
		static bool is_initialized() { return instance != nullptr; }
		static runtime& get_instance();

		~runtime();

		void startup();

		void shutdown() noexcept;

		void sync() noexcept;

		bool is_master_node() const { return local_nid == 0; }

		void broadcast_control_command(command_type cmd, const command_data& data);

	  private:
		static std::unique_ptr<runtime> instance;

		size_t num_nodes;
		node_id local_nid;

		uint64_t sync_id = 0;

		command_id next_control_command_id = command_id(1) << (std::numeric_limits<command_id::underlying_t>::digits - 1);

		std::unique_ptr<executor> exec;

		struct flush_handle {
			command_pkg pkg;
			std::vector<command_id> dependencies;
			MPI_Request req;
			mpi_support::single_use_data_type data_type;
		};
		std::deque<flush_handle> active_flushes;

		runtime(int* argc, char** argv[]);
		runtime(const runtime&) = delete;
		runtime(runtime&&) = delete;

		void flush_command(node_id target, const command_pkg& pkg, const std::vector<command_id>& dependencies);
	};

	std::unique_ptr<runtime> runtime::instance = nullptr;

	void runtime::init(int* argc, char** argv[]) {
		assert(!instance);
		instance = std::unique_ptr<runtime>(new runtime(argc, argv));
	}

	runtime& runtime::get_instance() {
		if(instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
		return *instance;
	}

	runtime::runtime(int* argc, char** argv[]) {
		int provided;
		MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
		assert(provided == MPI_THREAD_MULTIPLE);

		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		num_nodes = world_size;

		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		local_nid = world_rank;

		exec = std::make_unique<executor>(local_nid);
	}

	runtime::~runtime() {
		exec.reset();

		while(!active_flushes.empty()) {
			int done;
			MPI_Test(&active_flushes.begin()->req, &done, MPI_STATUS_IGNORE);
			if(done) { active_flushes.pop_front(); }
		}

		MPI_Finalize();
	}

	void runtime::startup() { exec->startup(); }

	void runtime::shutdown() noexcept {
		if(is_master_node()) { broadcast_control_command(command_type::SHUTDOWN, command_data{}); }

		exec->shutdown();
		instance.reset();
	}

	void runtime::sync() noexcept {
		sync_id++;

		if(is_master_node()) {
			sync_data cmd_data{};
			cmd_data.sync_id = sync_id;
			broadcast_control_command(command_type::SYNC, cmd_data);
		}

		while(exec->get_highest_executed_sync_id() < sync_id) {
			// std::this_thread::yield();
		}
	}

	void runtime::broadcast_control_command(command_type cmd, const command_data& data) {
		assert(is_master_node());
		for(auto n = 0u; n < num_nodes; ++n) {
			command_pkg pkg{next_control_command_id++, cmd, data};
			flush_command(n, pkg, {});
		}
	}

	void runtime::flush_command(node_id target, const command_pkg& pkg, const std::vector<command_id>& dependencies) {
		// Even though command packages are small enough to use a blocking send we want to be able to send to the master node as well,
		// which is why we have to use Isend after all. We also have to make sure that the buffer stays around until the send is complete.
		active_flushes.push_back(flush_handle{pkg, dependencies, MPI_REQUEST_NULL, {}});
		auto it = active_flushes.rbegin();
		it->data_type = mpi_support::build_single_use_composite_type(
		    {{sizeof(command_pkg), &it->pkg}, {sizeof(command_id) * dependencies.size(), it->dependencies.data()}});
		MPI_Isend(MPI_BOTTOM, 1, *it->data_type, static_cast<int>(target), mpi_support::TAG_CMD, MPI_COMM_WORLD, &active_flushes.rbegin()->req);

		// Cleanup finished transfers.
		// Just check the oldest flush. Since commands are small this will stay in equilibrium fairly quickly.
		int done;
		MPI_Test(&active_flushes.begin()->req, &done, MPI_STATUS_IGNORE);
		if(done) { active_flushes.pop_front(); }
	}


} // namespace detail
} // namespace celerity


//////////////////////////////////////////////////////////////////
///////////////////////// BENCHMARK //////////////////////////////
//////////////////////////////////////////////////////////////////

constexpr int warmup = 100;
constexpr int samples = 1000;
constexpr int serial_chain = 10;

[[gnu::noinline]] void reference_mpi_barrier() {
	MPI_Barrier(MPI_COMM_WORLD);
}

[[gnu::noinline]] void sync_only() {
	celerity::detail::runtime::get_instance().sync();
}

int size, rank;

template <typename Submit>
void benchmark(const char* name, Submit submit) {
	std::vector<uint64_t> sample_ns(samples);
	for(int i = 0; i < warmup; ++i) {
		submit();
	}
	for(int i = 0; i < samples; ++i) {
		const auto start = std::chrono::system_clock::now();
		submit();
		const auto end = std::chrono::system_clock::now();
		sample_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	}

	std::vector<uint64_t> all_ranks_sample_ns(rank == 0 ? samples * size : 0);
	MPI_Gather(sample_ns.data(), samples, MPI_UINT64_T, all_ranks_sample_ns.data(), samples, MPI_UINT64_T, 0, MPI_COMM_WORLD);
	if(rank == 0) {
		double mean = std::accumulate(all_ranks_sample_ns.begin(), all_ranks_sample_ns.end(), 0.0) / all_ranks_sample_ns.size();
		printf("%-55s %10.0f ns\n", name, mean);
		// fmt::print("{:55} {:10.0f} ns\n", name, mean);
		// fmt::print("{}", name);
		// for (int i = 0; i < all_ranks_sample_ns.size(); ++i) {
		//     fmt::print("{}{}", i % samples == 0 ? ';' : ',', all_ranks_sample_ns[i]);
		// }
		// fmt::print("\n");
	}
}

int main(int argc, char** argv) {
	celerity::detail::runtime::init(&argc, &argv);
	celerity::detail::runtime::get_instance().startup();

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	benchmark("MPI_Barrier", reference_mpi_barrier);
	benchmark("slow_full_sync", sync_only);

	celerity::detail::runtime::get_instance().shutdown();
}
