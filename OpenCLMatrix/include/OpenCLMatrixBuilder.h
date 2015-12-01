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
			vector<cl::Platform> platforms;
			vector<cl::Device> devices;

			cl::Platform::get(&platforms);
			platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
			for (const auto & device : devices) {
				info += device.getInfo<CL_DEVICE_NAME>().c_str();
				info += '\n';
			}

			cl::Context context{ devices };

			//build the program
			ifstream program_file{ "test2.cl" };
			string program_string(std::istreambuf_iterator<char>{program_file}, std::istreambuf_iterator<char>{});
			cl::Program::Sources source{ std::make_pair(program_string.c_str(), program_string.length() + 1) };
			cl::Program program{ context, source };
			program.build(devices);

			//put all the kernels into a map
			kernels.insert( std::make_pair("reduction_scalar", cl::Kernel{ program, "reduction_scalar" }));
		}

		unique_ptr<Matrix<T>> create(size_t rowCount, size_t columnCount) override {
			vector<vector<T>> data(rowCount);
			for (auto &data_row : data) {
				data_row.resize(columnCount);
			}
			unique_ptr<Matrix<T>> retval{ new OpenCLMatrix<T>{ data, kernels } };
			return retval;
		};

		unique_ptr<Matrix<T>> create(initializer_list<initializer_list<T>> lists) override {
			vector<vector<T>> data;
			for (auto list : lists) {
				data.push_back(vector<T>{list});
			}
			unique_ptr<Matrix<T>> retval{ new OpenCLMatrix<T>{ data, kernels } };
			return retval;
		};

		unique_ptr<Matrix<T>> create(const vector<vector<T>> &v) override {
			unique_ptr<Matrix<T>> retval{ new OpenCLMatrix<T>{ v, kernels } };
			return retval;
		};

		unique_ptr<Matrix<T>> createRowMatrix(const vector<T> &t) override {
			vector<vector<T>> data{ t };
			unique_ptr<Matrix<T>> retval{ new OpenCLMatrix<T>{ data, kernels } };
			return retval;
		}

		string getInfo() override {
			return "";
		};

	private:
		string info;
		unordered_map<string, cl::Kernel> kernels;
		vector<cl::Buffer> buffers;
	};
}
