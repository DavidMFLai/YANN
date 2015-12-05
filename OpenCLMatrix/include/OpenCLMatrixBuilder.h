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
		OpenCLMatrixBuilder<T>() {
			cl::Platform::get(&this->platforms);
			this->platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &this->devices);
			for (const auto & device : this->devices) {
				info += device.getInfo<CL_DEVICE_NAME>().c_str();
				info += '\n';
			}

			//create context
			this->context = cl::Context{ this->devices };

			//build the program
			ifstream program_file{ "test2.cl" };
			string program_string(std::istreambuf_iterator<char>{program_file}, std::istreambuf_iterator<char>{});
			cl::Program::Sources source{ std::make_pair(program_string.c_str(), program_string.length() + 1) };
			this->program = cl::Program{ this->context, source };
			this->program.build(this->devices);

			//put all the kernels into a map
			this->kernels.insert( std::make_pair("reduction_scalar", cl::Kernel{ this->program, "reduction_scalar" }));

			//create command queue
			this->queue = cl::CommandQueue{ this->context, this->devices[0], CL_QUEUE_PROFILING_ENABLE };
		}

		unique_ptr<Matrix<T>> create(size_t rowCount, size_t columnCount) override {
			cl::Buffer buffer{ context, CL_MEM_READ_WRITE, rowCount * columnCount * sizeof(T) };
			
			std::array<size_t, 2> dimensions{ rowCount, columnCount };

			unique_ptr<Matrix<T>> retval{ new OpenCLMatrix<T>{ std::move(buffer), std::move(dimensions), this->kernels, this->queue } };
			return retval;
		};

		unique_ptr<Matrix<T>> create(initializer_list<initializer_list<T>> lists) override {
			//total number of elements
			size_t number_of_elements = lists.size() * lists.begin()->size();

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

			unique_ptr<Matrix<T>> retval{ new OpenCLMatrix<T>{ std::move(buffer), std::move(dimensions), this->kernels, this->queue } };
			return retval;
		};

		unique_ptr<Matrix<T>> create(const vector<vector<T>> &v) override {
			//total number of elements
			size_t number_of_elements = v.size() * v.begin()->size();

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

			unique_ptr<Matrix<T>> retval{ new OpenCLMatrix<T>{ std::move(buffer), std::move(dimensions), this->kernels, this->queue } };
			return retval;
		};

		unique_ptr<Matrix<T>> createRowMatrix(const vector<T> &v) override {
			//total number of elements
			size_t number_of_elements = v.size();

			//prepare data for host ptr
			auto data = std::vector<T>{ v };

			std::array<size_t, 2> dimensions{ 1, v.size() };
			cl::Buffer buffer{ context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, number_of_elements * sizeof(T), const_cast<T*>(&data[0]) };

			unique_ptr<Matrix<T>> retval{ new OpenCLMatrix<T>{ std::move(buffer), std::move(dimensions), this->kernels, this->queue } };
			return retval;
		}

		string getInfo() override {
			return "OpenCLMatrix";
		};

	private:
		string info;
		vector<cl::Platform> platforms;
		vector<cl::Device> devices;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Context context;
		unordered_map<string, cl::Kernel> kernels;
	};
}
