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
		size_t saturation_workitem_count;
	};

	using std::vector;
	using std::unordered_map;
	using std::unique_ptr;
	using std::initializer_list;
	using std::string;
	using std::ifstream;

	template<typename T>
	class OpenCLMatrixBuilder : public MatrixBuilder<T> {
	private:
		static void setup_opencl_objects_and_shared_buffers(
			const std::string &kernels_full_path,
			const size_t &max_matrix_element_count,
			const cl_device_type device_type,
			const std::string &platform_name_contains,
			cl::Device &cl_device,
			cl::Context &context,
			cl::CommandQueue &queue,
			unordered_map<string, KernelWrapper> &kernel_wrappers,
			std::vector<cl::Buffer> &shared_scratch_buffer
			)
		{
			//get platform
			vector<cl::Platform> platforms;
			cl::Platform::get(&platforms);

			//find first device with OpenCL 2.0
			for (auto &platform : platforms) {
				if (platform.getInfo<CL_PLATFORM_NAME>().find(platform_name_contains) == string::npos) {
					//cannot find platform_name_contains within platform name, skipping
					continue;
				}
				vector<cl::Device> all_devices;
				try {
					platform.getDevices(device_type, &all_devices);
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
			auto program = cl::Program{ context, source };
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

			//create shared_scratch_buffers
			shared_scratch_buffer.emplace_back(context, CL_MEM_READ_WRITE, max_matrix_element_count * sizeof(T), nullptr, nullptr);
			shared_scratch_buffer.emplace_back(context, CL_MEM_READ_WRITE, max_matrix_element_count * sizeof(T), nullptr, nullptr);
		}

	public:
		OpenCLMatrixBuilder<T>() 
			: max_matrix_element_count{ 10000 },
			kernels_full_path{ KERNEL_FULL_PATH },
			device_info{ 2048 },
			platform_name_contains{"AMD"},
			device_type{ CL_DEVICE_TYPE_GPU },
			has_setup_opencl_objects{false}
		{}

		//default destructor
		~OpenCLMatrixBuilder() = default;

		//sets the number of workitems needed to saturate the OpenCL device. This number is used solely for optimization. 
		OpenCLMatrixBuilder &set_saturation_workitem_count(size_t saturation_workitem_count) {
			device_info.saturation_workitem_count = saturation_workitem_count;
			return *this;
		}

		//sets the device type to find when building an opencl matrix
		OpenCLMatrixBuilder &set_device_type(cl_device_type device_type) {
			this->device_type = device_type;
			return *this;
		}

		//sets the device type to find when building an opencl matrix
		OpenCLMatrixBuilder &set_platform_name_contains(const std::string &platform_name_contains) {
			this->platform_name_contains = platform_name_contains;
			return *this;
		}

		//sets kernels_full_path
		OpenCLMatrixBuilder &set_kernels_full_path(const std::string &kernels_full_path) {
			this->kernels_full_path = kernels_full_path;
			return *this;
		}

		//sets max matrix element count
		OpenCLMatrixBuilder &set_max_matrix_element_count(size_t max_matrix_element_count) {
			this->max_matrix_element_count = max_matrix_element_count;
			return *this;
		}


		unique_ptr<Matrix<T>> create(size_t rowCount, size_t columnCount) override {
			if (!has_setup_opencl_objects) {
				has_setup_opencl_objects = true;
				setup_opencl_objects_and_shared_buffers(kernels_full_path,
					max_matrix_element_count,
					device_type,
					platform_name_contains,
					cl_device,
					context,
					queue,
					kernel_wrappers,
					shared_scratch_buffer);
			}

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
			if (!has_setup_opencl_objects) {
				has_setup_opencl_objects = true;
				setup_opencl_objects_and_shared_buffers(kernels_full_path,
					max_matrix_element_count,
					device_type,
					platform_name_contains,
					cl_device,
					context,
					queue,
					kernel_wrappers,
					shared_scratch_buffer);
			}

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
			if (!has_setup_opencl_objects) {
				has_setup_opencl_objects = true;
				setup_opencl_objects_and_shared_buffers(kernels_full_path,
					max_matrix_element_count,
					device_type,
					platform_name_contains,
					cl_device,
					context,
					queue,
					kernel_wrappers,
					shared_scratch_buffer);
			}

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
			if (!has_setup_opencl_objects) {
				has_setup_opencl_objects = true;
				setup_opencl_objects_and_shared_buffers(kernels_full_path,
					max_matrix_element_count,
					device_type,
					platform_name_contains,
					cl_device,
					context,
					queue,
					kernel_wrappers,
					shared_scratch_buffer);
			}

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

		//set by setters
		DeviceInfo device_info;
		cl_device_type device_type;
		std::string platform_name_contains;
		string kernels_full_path;
		size_t max_matrix_element_count;

		//used to ensure that opencl boiler plate variables are generated only once
		bool has_setup_opencl_objects;

		//opencl variables that are created only once
		cl::Device cl_device;
		cl::CommandQueue queue;
		cl::Context context;
		unordered_map<string, KernelWrapper> kernel_wrappers;
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
