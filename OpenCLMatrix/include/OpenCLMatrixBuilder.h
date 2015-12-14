#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <unordered_map>

#define __CL_ENABLE_EXCEPTIONS
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif

#include "OpenCLMatrix.h"
#include "MatrixBuilder.h"

class KernelWrapper {
public:
	cl::Kernel clKernel;
	size_t kernel_work_group_size;
};

namespace {
	using std::vector;
	using std::unordered_map;
	using std::unique_ptr;
	using std::initializer_list;
	using std::string;
	using std::ifstream;

	template<typename T>
	class OpenCLMatrixBuilder : public MatrixBuilder<T> {
	public:
		OpenCLMatrixBuilder<T>(size_t max_matrix_element_count)
			: max_matrix_element_count{ max_matrix_element_count } {
			//get platform
			cl::Platform::get(&this->platforms);

			//find first device with OpenCL 2.0
			for (auto &platform : this->platforms) {
				vector<cl::Device> all_devices;
				platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
				for (auto &device : all_devices) {
					auto major_version_as_string = device.getInfo<CL_DEVICE_VERSION>().substr(7, 1);
					if (major_version_as_string == "2") {
						this->devices.push_back(device);
						break;
					}
				}
				if (this->devices.size() == 1) {
					break;
				}
			}

			//create context
			this->context = cl::Context{ this->devices };

			//build the program
			ifstream program_file{ "test2.cl" };
			string program_string(std::istreambuf_iterator<char>{program_file}, std::istreambuf_iterator<char>{});
			cl::Program::Sources source{ std::make_pair(program_string.c_str(), program_string.length() + 1) };
			this->program = cl::Program{ this->context, source };
			try {
				char options[] = "-cl-std=CL2.0 -g -s \"test2.cl\""; //see https://software.intel.com/en-us/node/539339
				this->program.build(this->devices, options);
			}
			catch (cl::Error e) {
				auto buildInfo = this->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->devices.at(0));
				std::cout << e.what() << '\n';
				std::cout << buildInfo << '\n';
			}

			//put all the kernels into a map
			OpenCLMatrixBuilder::add_to_wrapper(this->kernel_wrappers, "used_by_set_to_sum_of_rows", this->program, this->devices[0]);
			OpenCLMatrixBuilder::add_to_wrapper(this->kernel_wrappers, "set_to_sum_of", this->program, this->devices[0]);
			OpenCLMatrixBuilder::add_to_wrapper(this->kernel_wrappers, "per_row_multiply_reduction", this->program, this->devices[0]);
			OpenCLMatrixBuilder::add_to_wrapper(this->kernel_wrappers, "per_column_multiply_and_then_scale", this->program, this->devices[0]);
			OpenCLMatrixBuilder::add_to_wrapper(this->kernel_wrappers, "row_vectors_per_element_multiply_and_then_scale", this->program, this->devices[0]);
			OpenCLMatrixBuilder::add_to_wrapper(this->kernel_wrappers, "copy", this->program, this->devices[0]);
			OpenCLMatrixBuilder::add_to_wrapper(this->kernel_wrappers, "outer_product", this->program, this->devices[0]);

			//create command queue
			this->queue = cl::CommandQueue{ this->context, this->devices[0], CL_QUEUE_PROFILING_ENABLE };

			//auto local_mem_size = this->devices[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

			//create 2 scratch_buffers
			this->shared_scratch_buffer.emplace_back( this->context, CL_MEM_READ_WRITE, max_matrix_element_count * sizeof(T), nullptr, nullptr );
			this->shared_scratch_buffer.emplace_back( this->context, CL_MEM_READ_WRITE, max_matrix_element_count * sizeof(T), nullptr, nullptr );
		}

		OpenCLMatrixBuilder<T>()
			: OpenCLMatrixBuilder<T>(10000)
		{}

		unique_ptr<Matrix<T>> create(size_t rowCount, size_t columnCount) override {
			if (rowCount * columnCount > this->max_matrix_element_count) {
				throw CannotCreateError{ "rowCount * columnCount is greater than max_matrix_element_count" };
			}

			cl::Buffer buffer{ context, CL_MEM_READ_WRITE, rowCount * columnCount * sizeof(T) };
			
			std::array<size_t, 2> dimensions{ rowCount, columnCount };

			unique_ptr<Matrix<T>> retval{
				new OpenCLMatrix<T>{ 
					std::move(buffer),
					std::move(dimensions),
					this->kernel_wrappers,
					this->queue,
					this->shared_scratch_buffer
				}
			};
			return retval;
		};

		unique_ptr<Matrix<T>> create(initializer_list<initializer_list<T>> lists) override {
			//total number of elements
			size_t number_of_elements = lists.size() * lists.begin()->size();

			if (number_of_elements > this->max_matrix_element_count) {
				throw CannotCreateError{ "number_of_elements is greater than max_matrix_element_count" };
			}

			//prepare data for host ptr
			auto data = std::vector<T>();
			data.reserve(number_of_elements);
			for (auto list : lists) {
				for (auto item : list) {
					data.push_back(item);
				}
			}

			std::array<size_t, 2> dimensions{ lists.size(), lists.begin()->size() };
			cl::Buffer buffer{ context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, number_of_elements * sizeof(T), &data[0] };

			unique_ptr<Matrix<T>> retval{ 
				new OpenCLMatrix<T>{ 
					std::move(buffer), 
					std::move(dimensions), 
					this->kernel_wrappers, 
					this->queue, 
					this->shared_scratch_buffer
				} 
			};
			return retval;
		};

		unique_ptr<Matrix<T>> create(const vector<vector<T>> &v) override {
			//total number of elements
			size_t number_of_elements = v.size() * v.begin()->size();

			if (number_of_elements > this->max_matrix_element_count) {
				throw CannotCreateError{ "number_of_elements is greater than max_matrix_element_count" };
			}

			//prepare data for host ptr
			auto data = std::vector<T>();
			data.reserve(number_of_elements);
			for (auto list : v) {
				for (auto item : list) {
					data.push_back(item);
				}
			}

			std::array<size_t, 2> dimensions{ v.size(), v.begin()->size() };
			cl::Buffer buffer{ context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, number_of_elements * sizeof(T), &data[0] };

			unique_ptr<Matrix<T>> retval{ 
				new OpenCLMatrix<T>{ 
					std::move(buffer), 
					std::move(dimensions), 
					this->kernel_wrappers, 
					this->queue, 
					this->shared_scratch_buffer
				} 
			};
			return retval;
		};

		unique_ptr<Matrix<T>> createRowMatrix(const vector<T> &v) override {
			//total number of elements
			size_t number_of_elements = v.size();

			if (number_of_elements > this->max_matrix_element_count) {
				throw CannotCreateError{ "number_of_elements is greater than max_matrix_element_count" };
			}

			//prepare data for host ptr
			auto data = std::vector<T>{ v };

			std::array<size_t, 2> dimensions{ 1, v.size() };
			cl::Buffer buffer{ context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, number_of_elements * sizeof(T), const_cast<T*>(&data[0]) };

			unique_ptr<Matrix<T>> retval{ 
				new OpenCLMatrix<T>{ 
					std::move(buffer), 
					std::move(dimensions), 
					this->kernel_wrappers, 
					this->queue,
					this->shared_scratch_buffer
				}
			};
			return retval;
		}

		string getInfo() override {
			return "OpenCLMatrix";
		};

	private:
		static void checkOpenCLVersion(const cl::Platform &platform, int minimum_major_version)
		{
			auto major_version_as_string = platform.getInfo<CL_PLATFORM_VERSION>().substr(7, 1);
			if (stoi(major_version_as_string) < minimum_major_version) {
				throw std::exception("OpenCL version must be at least 2.0");
			}
		}

		static void add_to_wrapper(std::unordered_map<std::string, KernelWrapper> &kernel_wrappers,
			const string &kernel_name, 
			const cl::Program &program,
			const cl::Device &device) 
		{
			KernelWrapper kernel_wrapper;
			kernel_wrapper.clKernel = cl::Kernel{ program, kernel_name.c_str() };
			kernel_wrapper.kernel_work_group_size = kernel_wrapper.clKernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
			kernel_wrappers.insert(std::make_pair(kernel_name, kernel_wrapper));
		}

		string info;
		vector<cl::Platform> platforms;
		vector<cl::Device> devices;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Context context;
		unordered_map<string, KernelWrapper> kernel_wrappers;
		size_t max_matrix_element_count;
		std::vector<cl::Buffer> shared_scratch_buffer;
	};


	class CannotCreateError : public std::runtime_error {
	public:
		CannotCreateError(const char *str)
			: std::runtime_error(str)
		{};

		CannotCreateError(std::string str)
			: std::runtime_error(str)
		{};
	};
}
