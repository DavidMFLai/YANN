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
		OpenCLMatrix(cl::Buffer buffer, std::array<size_t, 2> dimensions,
			unordered_map<string, cl::Kernel> &kernels,
			cl::CommandQueue &command_queue,
			size_t max_work_group_size,
			std::vector<cl::Buffer> shared_scratch_buffer)
			: buffer{ buffer, {} }
			, kernels{ kernels }
			, command_queue{ command_queue }
			, max_work_group_size{ max_work_group_size }
			, shared_scratch_buffer{ shared_scratch_buffer }
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

		void subtract_by(const Matrix<T> &) override {
			return; //todo: implement
		}
		void set_to_sum_of_rows(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);

			//copy input buffer to 0th scratch buffer
			command_queue.enqueueCopyBuffer(input_cl.buffer.cl_buffer, 
				this->shared_scratch_buffer[0],
				0, 
				0,
				input.getColumnCount() * input.getRowCount() * sizeof(T)
				);

			//For an M-rows by N-columns data, we divide the data into max_work_group_size-columns by 1-row chunks
			//if M is not divisible by max_work_group_size, then the last workgroups will have (M % max_work_group_size) chunks
			size_t number_of_rows_at_input = input.getRowCount();
			while(number_of_rows_at_input > 1)
			{
				auto &kernel = kernels.at("used_by_set_to_sum_of_rows");
				kernel.setArg(0, this->shared_scratch_buffer[0]);
				kernel.setArg(1, this->max_work_group_size*sizeof(T), nullptr);

				cl::NDRange global_size{ input.getColumnCount(), number_of_rows_at_input };
				cl::NDRange local_size;
				if (number_of_rows_at_input < this->max_work_group_size) {
					local_size = cl::NDRange{ 1, number_of_rows_at_input };
				}
				else {
					local_size = cl::NDRange{ 1, this->max_work_group_size };
				}

				command_queue.enqueueNDRangeKernel(kernel, cl::NDRange{ 0, 0 }, global_size, local_size);

				number_of_rows_at_input = number_of_rows_at_input / this->max_work_group_size
					+ (input.getColumnCount() % this->max_work_group_size == 0) ? 0 : 1;

			}

			//copy 0th scratch buffer to this
			command_queue.enqueueCopyBuffer(this->shared_scratch_buffer[0],
				this->buffer.cl_buffer,
				0,
				0,
				input.getColumnCount() * sizeof(T)
				);

			return;
		}
		void set_to_sum_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			return;
		}
		void set_to_difference_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			return;
		}
		void set_to_product_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
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
	private:
		void copy_data_to_host() const {
			size_t number_of_elements = this->getRowCount() * this->getColumnCount();
			buffer.cl_buffer_mirror_on_host.resize(number_of_elements);
			command_queue.enqueueReadBuffer(buffer.cl_buffer, CL_BLOCKING, 0, number_of_elements * sizeof(T), &buffer.cl_buffer_mirror_on_host[0], nullptr, nullptr);
		}

	public:
		std::vector<T> &getElems() override {
			copy_data_to_host();
			return buffer.cl_buffer_mirror_on_host;
		}
		const std::vector<T> &getElems() const override {
			copy_data_to_host();
			return buffer.cl_buffer_mirror_on_host;
		}
		void zero() override {
			return;
		}

	private:
		size_t max_work_group_size;
		unordered_map<string, cl::Kernel> &kernels;
		cl::CommandQueue &command_queue;
		struct Buffer {
			cl::Buffer cl_buffer;
			mutable std::vector<T> cl_buffer_mirror_on_host;
		} buffer;
		std::vector<cl::Buffer> shared_scratch_buffer;
	};
}

