#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <unordered_map>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200 //opencl 2.0
#ifdef MAC
#include <OpenCL/cl2.h>
#else
#include <CL/cl2.hpp>
#endif

#include "OpenCLMatrix.h"
#include "MatrixBuilder.h"

namespace {

	struct KernelWrapper {
		cl::Kernel clKernel;
		size_t kernel_work_group_size;
	};

	struct DeviceInfo {
		cl_int saturation_workitem_count;
	};

	using std::vector;
	using std::unordered_map;
	using std::unique_ptr;
	using std::initializer_list;
	using std::string;
	using std::ifstream;

	template<typename T>
	class OpenCLMatrixBuilder : public MatrixBuilder<T> {
	public:
		OpenCLMatrixBuilder<T>(size_t max_matrix_element_count, const string &kernels_full_path)
			: max_matrix_element_count{ max_matrix_element_count },
			kernels_full_path{ kernels_full_path },
			device_info{2048} {
			//get platform
			cl::Platform::get(&platforms);

			//find first device with OpenCL 2.0
			for (auto &platform : platforms) {
				vector<cl::Device> all_devices;
				try {
					platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
				}
				catch (cl::Error e) {
					//thrown due to it cant find devices of the selected type. There is no way of finding the number of devices in OpenCL's C++ Wrapper.
				}
				for (auto &device : all_devices) {
					auto major_version_as_string = device.getInfo<CL_DEVICE_VERSION>().substr(7, 1);
					if (major_version_as_string == "2") {
						cl_device = device;
						break;
					}
				}
			}

			//create context
			context = cl::Context{ cl_device };

			//build the program
			ifstream program_file{ kernels_full_path };
			string program_string(std::istreambuf_iterator<char>{program_file}, std::istreambuf_iterator<char>{});
			cl::Program::Sources source{ program_string };
			program = cl::Program{ context, source };
			try {
				char options[] = "-cl-std=CL2.0";
				//char options[] = "-cl-std=CL2.0 -g -s \"include/test2.cl\""; //see https://software.intel.com/en-us/node/539339
				auto one_device_in_vector = std::vector<cl::Device>{ cl_device };
				program.build(one_device_in_vector, options);
			}
			catch (cl::Error e) {
				auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_device);
				std::cout << e.what() << '\n';
				std::cout << buildInfo << '\n';
				throw e;
			}

			//Set kernel_wrappers, which entry contains the kernel itself with other auxilary information
			OpenCLMatrixBuilder<T>::set_kernel_wrappers(kernel_wrappers, program, cl_device);

			//create command queue
			queue = cl::CommandQueue{ context, cl_device, CL_QUEUE_PROFILING_ENABLE };

			//create 2 scratch_buffers
			shared_scratch_buffer.emplace_back( context, CL_MEM_READ_WRITE, max_matrix_element_count * sizeof(T), nullptr, nullptr );
			shared_scratch_buffer.emplace_back( context, CL_MEM_READ_WRITE, max_matrix_element_count * sizeof(T), nullptr, nullptr );
		}

		OpenCLMatrixBuilder<T>()
			: OpenCLMatrixBuilder<T>(10000, KERNEL_FULL_PATH)
		{}

		//default destructor
		~OpenCLMatrixBuilder() = default;

		OpenCLMatrixBuilder &set_alu_count(size_t saturation_workitem_count) {
			device_info.saturation_workitem_count = saturation_workitem_count;
			return *this;
		}
		
		unique_ptr<Matrix<T>> create(size_t rowCount, size_t columnCount) override {
			if (rowCount * columnCount > max_matrix_element_count) {
				throw CannotCreateError{ "rowCount * columnCount is greater than max_matrix_element_count" };
			}

			cl::Buffer buffer{ context, CL_MEM_READ_WRITE, rowCount * columnCount * sizeof(T) };
			
			std::array<size_t, 2> dimensions{ rowCount, columnCount };

			unique_ptr<Matrix<T>> retval = std::make_unique<OpenCLMatrix<T>>(
				std::move(buffer),
				std::move(dimensions),
				kernel_wrappers,
				device_info,
				queue,
				shared_scratch_buffer);
			return retval;
		};

		unique_ptr<Matrix<T>> create(initializer_list<initializer_list<T>> lists) override {
			//total number of elements
			size_t number_of_elements = lists.size() * lists.begin()->size();

			if (number_of_elements > max_matrix_element_count) {
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
					kernel_wrappers, 
					device_info,
					queue, 
					shared_scratch_buffer
				} 
			};
			return retval;
		};

		unique_ptr<Matrix<T>> create(const vector<vector<T>> &v) override {
			//total number of elements
			size_t number_of_elements = v.size() * v.begin()->size();

			if (number_of_elements > max_matrix_element_count) {
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
					kernel_wrappers, 
					device_info,
					queue, 
					shared_scratch_buffer
				} 
			};
			return retval;
		};

		unique_ptr<Matrix<T>> createRowMatrix(const vector<T> &v) override {
			//total number of elements
			size_t number_of_elements = v.size();

			if (number_of_elements > max_matrix_element_count) {
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
					kernel_wrappers, 
					device_info,
					queue,
					shared_scratch_buffer
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

		static void set_kernel_wrappers(unordered_map<string, KernelWrapper> &kernel_wrappers, cl::Program &program, const cl::Device &device)
		{
			kernel_wrappers.clear();

			//put all the kernels into a map
			vector<cl::Kernel> kernels;
			program.createKernels(&kernels);

			for (auto &kernel : kernels) {
				KernelWrapper kernel_wrapper;
				kernel_wrapper.clKernel = kernel;
				kernel_wrapper.kernel_work_group_size = kernel_wrapper.clKernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
				kernel_wrappers.insert(std::make_pair(kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(), kernel_wrapper));
			}
		}

		string info;
		vector<cl::Platform> platforms;
		cl::Device cl_device;
		DeviceInfo device_info;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Context context;
		unordered_map<string, KernelWrapper> kernel_wrappers;
		size_t max_matrix_element_count;
		string kernels_full_path;
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
