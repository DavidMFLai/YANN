#pragma once

#define NOMINMAX
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200 //opencl 2.0

#ifdef MAC
#include <OpenCL/cl2.h>
#else
#include <CL/cl2.hpp>
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
	struct Buffer {
		cl::Buffer cl_buffer;
		mutable std::vector<T> cl_buffer_mirror_on_host;
	};

	template<typename T>
	class OpenCLMatrix : public Matrix<T> {
	public:
		//Sole constructor
		OpenCLMatrix(cl::Buffer buffer,
			std::array<size_t, 2> dimensions,
			const unordered_map<string, KernelWrapper> &kernel_wrappers,
			const DeviceInfo &device_info,
			const cl::CommandQueue &command_queue,
			const std::vector<cl::Buffer> &shared_scratch_buffer)
			: buffer{ buffer,{} }
			, kernel_wrappers{ kernel_wrappers }
			, device_info{device_info}
			, command_queue{ command_queue }
			, shared_scratch_buffer{ shared_scratch_buffer }
		{
			matrixAccessProperties.setDimensions(dimensions[0], dimensions[1]);
		};

		//deleted default, copy, move, copy assignment, and move assignment
		OpenCLMatrix() = delete;
		OpenCLMatrix(OpenCLMatrix &&) = delete;
		OpenCLMatrix(const OpenCLMatrix &) = delete;
		OpenCLMatrix &operator=(OpenCLMatrix &&) = delete;
		OpenCLMatrix &operator=(const OpenCLMatrix &) = delete;

		//default destructor
		~OpenCLMatrix() = default;

	private:
		void subtract_by(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("subtract_by").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("subtract_by").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, input_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength(), this->getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min(this->getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		void set_to_sum_of_rows(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);

			//copy input buffer to 0th scratch buffer
			command_queue.enqueueCopyBuffer(input_cl.buffer.cl_buffer, 
				shared_scratch_buffer[0],
				0, 
				0,
				input.getRowLength() * input.getColumnLength() * sizeof(T)
				);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("set_to_sum_of_rows").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("set_to_sum_of_rows").clKernel;
			
			//For an M-rows by N-columns data, we divide the data into max_work_group_size-columns by 1-row chunks
			//if M is not divisible by max_work_group_size, then the last workgroups will have (M % max_work_group_size) chunks
			size_t column_length_at_input = input.getColumnLength();
			while(column_length_at_input > 1)	{
				//Get clKernel and set arguments 
				clKernel.setArg(0, shared_scratch_buffer[0]);
				cl::LocalSpaceArg arg = cl::Local(max_work_group_size*sizeof(T));
				clKernel.setArg(1, arg);

				//enqueueNDRangeKernel 
				cl::NDRange global_size{ input.getRowLength(), column_length_at_input };
				cl::NDRange local_size = cl::NDRange{ 1, std::min(column_length_at_input, max_work_group_size) };
				command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);

				//calculate new column length after execution
				{
					bool has_odd_column = (column_length_at_input % max_work_group_size) == 0 ? 0 : 1;
					column_length_at_input = column_length_at_input / max_work_group_size + has_odd_column;
				}
			}

			//copy 0th scratch buffer to this
			command_queue.enqueueCopyBuffer(shared_scratch_buffer[0],
				buffer.cl_buffer,
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
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, lhs_cl.buffer.cl_buffer);
			clKernel.setArg(2, rhs_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			size_t number_of_elements = lhs_cl.getRowLength() * lhs_cl.getColumnLength();
			cl::NDRange global_size{ number_of_elements };
			cl::NDRange local_size = cl::NDRange{ std::min(number_of_elements, max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		void set_to_difference_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			const OpenCLMatrix &lhs_cl = dynamic_cast<const OpenCLMatrix &>(lhs);
			const OpenCLMatrix &rhs_cl = dynamic_cast<const OpenCLMatrix &>(rhs);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("set_to_difference_of").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("set_to_difference_of").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, lhs_cl.buffer.cl_buffer);
			clKernel.setArg(2, rhs_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			size_t number_of_elements = lhs_cl.getRowLength() * lhs_cl.getColumnLength();
			cl::NDRange global_size{ number_of_elements };
			cl::NDRange local_size = cl::NDRange{ std::min(number_of_elements, max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		static void run_set_to_product_of_where_lhs_is_a_long_row_matrix(
			cl::Buffer &results_buffer,
			const OpenCLMatrix &lhs_cl,
			const OpenCLMatrix &rhs_cl,
			std::unordered_map<std::string, KernelWrapper> &kernel_wrappers,
			const cl::CommandQueue &command_queue
			) 
		{
			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("set_to_product_of_where_lhs_is_a_long_row_matrix").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("set_to_product_of_where_lhs_is_a_long_row_matrix").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, results_buffer);
			clKernel.setArg(1, static_cast<unsigned int>(lhs_cl.getRowLength()));
			clKernel.setArg(2, static_cast<unsigned int>(rhs_cl.getRowLength()));
			clKernel.setArg(3, lhs_cl.buffer.cl_buffer);
			clKernel.setArg(4, rhs_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ rhs_cl.getRowLength() };
			cl::NDRange local_size = cl::NDRange{ std::min(rhs_cl.getRowLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		static void run_per_row_multiply(
			cl::Buffer &results_buffer,
			const OpenCLMatrix &multipliers_cl,
			const OpenCLMatrix &multiplicand_cl,
			std::unordered_map<std::string, KernelWrapper> &kernel_wrappers,
			const cl::CommandQueue &command_queue) 
		{
			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_row_multiply").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_row_multiply").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, results_buffer);
			clKernel.setArg(1, multipliers_cl.buffer.cl_buffer);
			clKernel.setArg(2, multiplicand_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ multiplicand_cl.getRowLength(), multiplicand_cl.getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min(multiplicand_cl.getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		//inout_buffer stores both the input and the results. results_buffer is copied from
		static void run_set_to_sum_of_rows(
			cl::Buffer &inout_buffer,
			size_t row_length,
			size_t column_length,
			std::unordered_map<std::string, KernelWrapper> &kernel_wrappers,
			const cl::CommandQueue &command_queue)
		{
			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("set_to_sum_of_rows").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("set_to_sum_of_rows").clKernel;

			//For an M-rows by N-columns data, we divide the data into max_work_group_size-columns by 1-row chunks
			//if M is not divisible by max_work_group_size, then the last workgroups will have (M % max_work_group_size) chunks
			size_t column_length_at_input = column_length;
			while (column_length_at_input > 1) {
				//Get clKernel and set arguments 
				clKernel.setArg(0, inout_buffer);
				cl::LocalSpaceArg arg = cl::Local(max_work_group_size*sizeof(T));
				clKernel.setArg(1, arg);

				//enqueueNDRangeKernel 
				cl::NDRange global_size{ row_length, column_length_at_input };
				cl::NDRange local_size = cl::NDRange{ 1, std::min(column_length_at_input, max_work_group_size) };
				command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);

				//calculate new column length after execution
				{
					bool has_odd_column = (column_length_at_input % max_work_group_size) == 0 ? 0 : 1;
					column_length_at_input = column_length_at_input / max_work_group_size + has_odd_column;
				}
			}
		}

		static void run_set_to_product_of(
			cl::Buffer &results_buffer,
			size_t M,
			size_t K,
			size_t N,
			const cl::Buffer &lhs_cl_buffer,
			const cl::Buffer &rhs_cl_buffer,
			std::unordered_map<std::string, KernelWrapper> &kernel_wrappers,
			const cl::CommandQueue &command_queue)
		{
			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("set_to_product_of").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("set_to_product_of").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, results_buffer);
			clKernel.setArg(1, static_cast<unsigned int>(M));
			clKernel.setArg(2, static_cast<unsigned int>(K));
			clKernel.setArg(3, static_cast<unsigned int>(N));
			clKernel.setArg(4, lhs_cl_buffer);
			clKernel.setArg(5, rhs_cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ N, M };
			cl::NDRange local_size = cl::NDRange{ 1, std::min(M, max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		void set_to_product_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			const OpenCLMatrix &lhs_cl = dynamic_cast<const OpenCLMatrix &>(lhs);
			const OpenCLMatrix &rhs_cl = dynamic_cast<const OpenCLMatrix &>(rhs);

			if (lhs.getColumnLength() == 1) {
				//row matrix multiplied with a non-row matrix
				if(rhs.getRowLength() >= device_info.saturation_workitem_count) {
					run_set_to_product_of_where_lhs_is_a_long_row_matrix(buffer.cl_buffer, lhs_cl, rhs_cl, kernel_wrappers, command_queue);
				}
				else {
					run_per_row_multiply(shared_scratch_buffer[0], lhs_cl, rhs_cl, kernel_wrappers, command_queue);
					run_set_to_sum_of_rows(shared_scratch_buffer[0], rhs.getRowLength(), rhs.getColumnLength(), kernel_wrappers, command_queue);

					//copy shared_scratch_buffer[0] to this->buffer.cl_buffer
					command_queue.enqueueCopyBuffer(shared_scratch_buffer[0],
						buffer.cl_buffer,
						0,
						0,
						rhs.getRowLength() * sizeof(T)
						);
				}
			}
			else {
				run_set_to_product_of(buffer.cl_buffer,
					lhs.getColumnLength(),
					lhs.getRowLength(),
					rhs.getRowLength(),
					lhs_cl.buffer.cl_buffer,
					rhs_cl.buffer.cl_buffer,
					kernel_wrappers, 
					command_queue
					);
			}
		}

		void per_Element_Sigmoid(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_element_sigmoid").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_element_sigmoid").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, input_cl.buffer.cl_buffer);

			auto number_of_elements = this->getRowLength() * this->getColumnLength();
			cl::NDRange global_size{ number_of_elements };
			cl::NDRange local_size = cl::NDRange{ std::min(number_of_elements, max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		void per_Element_Sigmoid_Prime(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_element_sigmoid_prime").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_element_sigmoid_prime").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, input_cl.buffer.cl_buffer);

			auto number_of_elements = this->getRowLength() * this->getColumnLength();
			cl::NDRange global_size{ number_of_elements };
			cl::NDRange local_size = cl::NDRange{ std::min(number_of_elements, max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		void per_Element_Tanh(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_element_tanh").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_element_tanh").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, input_cl.buffer.cl_buffer);

			auto number_of_elements = this->getRowLength() * this->getColumnLength();
			cl::NDRange global_size{ number_of_elements };
			cl::NDRange local_size = cl::NDRange{ std::min(number_of_elements, max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		void per_Element_Tanh_Prime(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_element_tanh_prime").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_element_tanh_prime").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, input_cl.buffer.cl_buffer);

			auto number_of_elements = this->getRowLength() * this->getColumnLength();
			cl::NDRange global_size{ number_of_elements };
			cl::NDRange local_size = cl::NDRange{ std::min(number_of_elements, max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		void per_Column_Multiply_AndThen_Transpose(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) override {
			const OpenCLMatrix &multipliers_cl = dynamic_cast<const OpenCLMatrix &>(multipliers);
			const OpenCLMatrix &multiplicand_cl = dynamic_cast<const OpenCLMatrix &>(multiplicand);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_column_multiply_and_then_transpose").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_column_multiply_and_then_transpose").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, multipliers_cl.buffer.cl_buffer);
			clKernel.setArg(2, multiplicand_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ multiplicand.getRowLength(), multiplicand.getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min(multiplicand.getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		void per_Column_Multiply_AndThen_Scale(const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) override {
			const OpenCLMatrix &multipliers_cl = dynamic_cast<const OpenCLMatrix &>(multipliers);
			const OpenCLMatrix &multiplicand_cl = dynamic_cast<const OpenCLMatrix &>(multiplicand);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_column_multiply_and_then_scale").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_column_multiply_and_then_scale").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, multipliers_cl.buffer.cl_buffer);
			clKernel.setArg(2, multiplicand_cl.buffer.cl_buffer);
			clKernel.setArg(3, scale);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength(), this->getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min(this->getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		void per_Row_Multiply(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) override {
			const OpenCLMatrix &multipliers_cl = dynamic_cast<const OpenCLMatrix &>(multipliers);
			const OpenCLMatrix &multiplicand_cl = dynamic_cast<const OpenCLMatrix &>(multiplicand);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("per_row_multiply").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("per_row_multiply").clKernel;
			
			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, multipliers_cl.buffer.cl_buffer);
			clKernel.setArg(2, multiplicand_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength(), this->getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min(this->getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		void row_Vectors_Per_Element_Multiply_AndThen_Scale(const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) override {
			const OpenCLMatrix &row_vector_1_cl = dynamic_cast<const OpenCLMatrix &>(row_vector_1);
			const OpenCLMatrix &row_vector_2_cl = dynamic_cast<const OpenCLMatrix &>(row_vector_2);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("row_vectors_per_element_multiply_and_then_scale").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("row_vectors_per_element_multiply_and_then_scale").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, row_vector_1_cl.buffer.cl_buffer);
			clKernel.setArg(2, row_vector_2_cl.buffer.cl_buffer);
			clKernel.setArg(3, scale);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength() };
			cl::NDRange local_size = cl::NDRange{ std::min(this->getRowLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0 }, global_size, local_size);
		}

		void copy(const Matrix<T> &input) override {
			const OpenCLMatrix &input_cl = dynamic_cast<const OpenCLMatrix &>(input);
			command_queue.enqueueCopyBuffer(input_cl.buffer.cl_buffer, buffer.cl_buffer, 0, 0, this->getRowLength() * this->getColumnLength() * sizeof(T));
		}

		void outer_product(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
			assert(lhs.getColumnLength() == 1);
			assert(rhs.getColumnLength() == 1);
			assert(lhs.getRowLength() == this->getColumnLength());
			assert(rhs.getRowLength() == this->getRowLength());

			const OpenCLMatrix &lhs_cl = dynamic_cast<const OpenCLMatrix &>(lhs);
			const OpenCLMatrix &rhs_cl = dynamic_cast<const OpenCLMatrix &>(rhs);

			//get clKernel and its work group size for this device
			size_t max_work_group_size = kernel_wrappers.at("outer_product").kernel_work_group_size;
			auto &clKernel = kernel_wrappers.at("outer_product").clKernel;

			//Set arguments for clKernel
			clKernel.setArg(0, buffer.cl_buffer);
			clKernel.setArg(1, lhs_cl.buffer.cl_buffer);
			clKernel.setArg(2, rhs_cl.buffer.cl_buffer);

			//enqueueNDRangeKernel 
			cl::NDRange global_size{ this->getRowLength(), this->getColumnLength() };
			cl::NDRange local_size = cl::NDRange{ 1, std::min(this->getColumnLength(), max_work_group_size) };
			command_queue.enqueueNDRangeKernel(clKernel, cl::NDRange{ 0, 0 }, global_size, local_size);
		}

		void copy_from_vector(const std::vector<T> &input) override {
			command_queue.enqueueWriteBuffer(buffer.cl_buffer, 
				CL_BLOCKING, 
				0, 
				input.size() * sizeof(T), 
				&input[0]);
		}

		void copy_data_to_host() const {
			size_t number_of_elements = this->getColumnLength() * this->getRowLength();
			buffer.cl_buffer_mirror_on_host.resize(number_of_elements);
			command_queue.enqueueReadBuffer(buffer.cl_buffer, CL_BLOCKING, 0, number_of_elements * sizeof(T), &buffer.cl_buffer_mirror_on_host[0], nullptr, nullptr);
		}

	public:

		T &at(size_t i, size_t j) override {
			copy_data_to_host();
			return buffer.cl_buffer_mirror_on_host[matrixAccessProperties(i, j)];
		}

		const T &at(size_t i, size_t j) const override {
			copy_data_to_host();
			return buffer.cl_buffer_mirror_on_host[matrixAccessProperties(i, j)];
		}

		void zero() override {

			auto number_of_elements = this->getColumnLength() * this->getRowLength();

			auto zeroes_data = std::make_unique<T[]>(number_of_elements);

			command_queue.enqueueWriteBuffer(buffer.cl_buffer,
				CL_BLOCKING,
				0,
				number_of_elements * sizeof(T),
				&zeroes_data[0]);
		}

		std::vector<T> &getElems() override {
			copy_data_to_host();
			return buffer.cl_buffer_mirror_on_host;
		}

		const std::vector<T> &getElems() const override {
			copy_data_to_host();
			return buffer.cl_buffer_mirror_on_host;
		}

	private:
		std::unordered_map<std::string, KernelWrapper> kernel_wrappers;
		DeviceInfo device_info;
		cl::CommandQueue command_queue;
		Buffer<T> buffer;
		std::vector<cl::Buffer> shared_scratch_buffer;
	};
}

