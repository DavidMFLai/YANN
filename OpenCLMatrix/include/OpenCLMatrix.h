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
#include <algorithm>

namespace {
	using std::unordered_map;
	using std::vector;

	template<typename T>
	class OpenCLMatrix : public Matrix<T> {

	private:
		friend class OpenCLMatrixBuilder<T>;

		//Sole constructor
		OpenCLMatrix(cl::Buffer buffer, 
			std::array<size_t, 2> dimensions,
			unordered_map<string, KernelWrapper> &kernel_wrappers,
			cl::CommandQueue &command_queue,
			std::vector<cl::Buffer> shared_scratch_buffer)
			: buffer{ buffer, {} }
			, kernel_wrappers{ kernel_wrappers }
			, command_queue{ command_queue }
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
				input.getRowLength() * input.getColumnLength() * sizeof(T)
				);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("used_by_set_to_sum_of_rows").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("used_by_set_to_sum_of_rows").clKernel;
			
			//For an M-rows by N-columns data, we divide the data into max_work_group_size-columns by 1-row chunks
			//if M is not divisible by max_work_group_size, then the last workgroups will have (M % max_work_group_size) chunks
			size_t column_length_at_input = input.getColumnLength();
			while(column_length_at_input > 1)	{
				//Get clKernel and set arguments 
				clKernel.setArg(0, this->shared_scratch_buffer[0]);
				cl::LocalSpaceArg arg = cl::__local(max_work_group_size*sizeof(T));
				clKernel.setArg(1, arg);

				//enqueueNDRangeKernel 
				cl::NDRange global_size{ input.getRowLength(), column_length_at_input };
				cl::NDRange local_size = cl::NDRange{ 1, std::min<size_t>(column_length_at_input, max_work_group_size) };
				command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);

				//calculate new column length after execution
				{
					bool has_odd_column = (column_length_at_input % max_work_group_size) == 0 ? 0 : 1;
					column_length_at_input = column_length_at_input / max_work_group_size + has_odd_column;
				}
			}

			//copy 0th scratch buffer to this
			command_queue.enqueueCopyBuffer(this->shared_scratch_buffer[0],
				this->buffer.cl_buffer,
				0,
				0,
				input.getRowLength() * sizeof(T)
				);
		}

		void set_to_sum_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			const OpenCLMatrix &lhs_cl = dynamic_cast<const OpenCLMatrix &>(lhs);
			const OpenCLMatrix &rhs_cl = dynamic_cast<const OpenCLMatrix &>(rhs);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("set_to_sum_of").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("set_to_sum_of").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, this->buffer.cl_buffer);
			clKernel.setArg(1, lhs_cl.buffer.cl_buffer);
			clKernel.setArg(2, rhs_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			size_t number_of_elements = lhs_cl.getRowLength() * lhs_cl.getColumnLength();
			cl::NDRange global_size{ number_of_elements };
			cl::NDRange local_size = cl::NDRange{ std::min<size_t>(number_of_elements, max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
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
			const OpenCLMatrix &multipliers_cl = dynamic_cast<const OpenCLMatrix &>(multipliers);
			const OpenCLMatrix &multiplicand_cl = dynamic_cast<const OpenCLMatrix &>(multiplicand);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_column_multiply_and_then_scale").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_column_multiply_and_then_scale").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, this->buffer.cl_buffer);
			clKernel.setArg(1, multipliers_cl.buffer.cl_buffer);
			clKernel.setArg(2, multiplicand_cl.buffer.cl_buffer);
			clKernel.setArg(3, scale);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength(), this->getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min<size_t>(this->getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		void per_Row_Multiply(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) override {
			const OpenCLMatrix &multipliers_cl = dynamic_cast<const OpenCLMatrix &>(multipliers);
			const OpenCLMatrix &multiplicand_cl = dynamic_cast<const OpenCLMatrix &>(multiplicand);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_row_multiply_reduction").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_row_multiply_reduction").clKernel;
			
			//Set arguments for clKernel
			clKernel.setArg(0, this->buffer.cl_buffer);
			clKernel.setArg(1, multipliers_cl.buffer.cl_buffer);
			clKernel.setArg(2, multiplicand_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength(), this->getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min<size_t>(this->getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		void row_Vectors_Per_Element_Multiply_AndThen_Scale(const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) override {
			const OpenCLMatrix &row_vector_1_cl = dynamic_cast<const OpenCLMatrix &>(row_vector_1);
			const OpenCLMatrix &row_vector_2_cl = dynamic_cast<const OpenCLMatrix &>(row_vector_2);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("row_vectors_per_element_multiply_and_then_scale").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("row_vectors_per_element_multiply_and_then_scale").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, this->buffer.cl_buffer);
			clKernel.setArg(1, row_vector_1_cl.buffer.cl_buffer);
			clKernel.setArg(2, row_vector_2_cl.buffer.cl_buffer);
			clKernel.setArg(3, scale);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength() };
			cl::NDRange local_size = cl::NDRange{ std::min<size_t>(this->getRowLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		void copy(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("copy").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("copy").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, this->buffer.cl_buffer);
			clKernel.setArg(1, input_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength(), this->getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min<size_t>(this->getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
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
			size_t number_of_elements = this->getColumnLength() * this->getRowLength();
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
		std::unordered_map<std::string, KernelWrapper> &kernel_wrappers;
		cl::CommandQueue &command_queue;
		struct Buffer {
			cl::Buffer cl_buffer;
			mutable std::vector<T> cl_buffer_mirror_on_host;
		} buffer;
		std::vector<cl::Buffer> shared_scratch_buffer;
	};
}

