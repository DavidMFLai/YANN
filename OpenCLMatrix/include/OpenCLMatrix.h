#pragma once

#define __CL_ENABLE_EXCEPTIONS
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif


#include "Matrix.h"
#include "OpenCLMatrixBuilder.h"

#include <vector>
#include <unordered_map>

namespace {
	using std::unordered_map;
	using std::vector;

	template<typename T>
	class OpenCLMatrix : public Matrix<T> {

	private:
		friend class OpenCLMatrixBuilder<T>;

		//Only constructor
		OpenCLMatrix(cl::Buffer buffer, std::array<size_t, 2> dimensions, unordered_map<string, cl::Kernel> &kernels, cl::CommandQueue &command_queue)
			: buffer{ buffer }
			, kernels{ kernels }
			, command_queue{ command_queue }
		{
			this->matrixAccessProperties.setDimensions(dimensions[0], dimensions[1]);
		};

		//deleted default, copy, move, copy assignment, and move assignment
		OpenCLMatrix() = delete;
		OpenCLMatrix(OpenCLMatrix &&) = delete;
		OpenCLMatrix(const OpenCLMatrix &) = delete;
		OpenCLMatrix &operator=(OpenCLMatrix &&) = delete;
		OpenCLMatrix &operator=(const OpenCLMatrix &) = delete;

	private:
		bool is_equal(const Matrix<T> &) const override {
			return false; //todo: implement
		};

		void subtract_andThen_assign(const Matrix<T> &) override {
			return; //todo: implement
		}
		void sum_of_rows(const Matrix<T> &input) override {
			return;
		}
		void add(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			return;
		}
		void minus(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			return;
		}
		void multiply(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			return;
		}
		void per_Element_Sigmoid(const Matrix<T> &input) override {
			return;
		}
		void per_Element_Sigmoid_Prime(const Matrix<T> &sigmoid_value) override {
			return;
		}
		void per_Element_Tanh(const Matrix<T> &input) override {
			return;
		}
		void per_Element_Tanh_Prime(const Matrix<T> &tanh_value) override {
			return;
		}
		void per_Column_Multiply_AndThen_Transpose(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) override {
			return;
		}
		void per_Column_Multiply_AndThen_Scale(const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) override {
			return;
		}
		void per_Row_Multiply(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) override {
			return;
		}
		void row_Vectors_Per_Element_Multiply_AndThen_Scale(const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) override {
			return;
		}
		void copy(const Matrix<T> &input) override {
			return;
		}
		void outer_product(const Matrix<T> &input1, const Matrix<T> &input2) override {
			return;
		}
		void copy_from_vector(const std::vector<T> &input) override {
			return;
		}

	public:

		T temp;
		T &at(size_t i, size_t j) override {
			return temp;
		}
		const T &at(size_t i, size_t j) const override {
			return T{};
		}
		std::vector<T> &getElems() override {
			return std::vector<T>{};
		}
		const std::vector<T> &getElems() const override {
			return std::vector<T>{};
		}
		void zero() override {
			return;
		}

	private:
		unordered_map<string, cl::Kernel> &kernels;
		cl::CommandQueue &command_queue;
		cl::Buffer buffer;
	};
}

