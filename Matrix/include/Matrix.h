#pragma once

#include <array>
#include <cassert>
#include <vector>

namespace {
	template <typename T>
	class Matrix {
		template<typename U>
		friend bool operator==(const Matrix<U> &lhs, const Matrix<U> &rhs);

	public:
		//Getting dimensions
		std::array<size_t, 2> getDimensions() const {
			return this->matrixAccessProperties.dimensions;
		}

		//Getting No. of rows
		size_t getColumnLength() const {
			return this->matrixAccessProperties.dimensions[0];
		}

		//Getting No. of columns
		size_t getRowLength() const {
			return this->matrixAccessProperties.dimensions[1];
		}

	protected:
		struct MatrixAccessProperties {
			MatrixAccessProperties() = default;

			void setDimensions(size_t rowCount, size_t columnCount) {
				this->dimensions = { rowCount, columnCount };
			}

			size_t operator()(size_t i, size_t j) const {
				assert(i < this->dimensions[0]);
				assert(j < this->dimensions[1]);
				return i*this->dimensions[1] + j;
			}

			std::array<size_t, 2> dimensions;
		} matrixAccessProperties;

	private:
		virtual void subtract_by(const Matrix<T> &) = 0;
		virtual void set_to_sum_of_rows(const Matrix<T> &input) = 0;
		virtual void set_to_sum_of(const Matrix<T> &lhs, const Matrix<T> &rhs) = 0;
		virtual void set_to_difference_of(const Matrix<T> &lhs, const Matrix<T> &rhs) = 0;
		virtual void set_to_product_of(const Matrix<T> &lhs, const Matrix<T> &rhs) = 0;
		virtual void per_Element_Sigmoid(const Matrix<T> &input) = 0;
		virtual void per_Element_Sigmoid_Prime(const Matrix<T> &sigmoid_value) = 0;
		virtual void per_Element_Tanh(const Matrix<T> &input) = 0;
		virtual void per_Element_Tanh_Prime(const Matrix<T> &tanh_value) = 0;
		virtual void per_Column_Multiply_AndThen_Transpose(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) = 0;
		virtual void per_Column_Multiply_AndThen_Scale(const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) = 0;
		virtual void per_Row_Multiply(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) = 0;
		virtual void row_Vectors_Per_Element_Multiply_AndThen_Scale(const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) = 0;
		virtual void copy(const Matrix<T> &input) = 0;
		virtual void outer_product(const Matrix<T> &input1, const Matrix<T> &input2) = 0;
		virtual void copy_from_vector(const std::vector<T> &input) = 0;

	public:
		virtual T &at(size_t i, size_t j) = 0;
		virtual const T &at(size_t i, size_t j) const = 0;
		virtual std::vector<T> &getElems() = 0;
		virtual const std::vector<T> &getElems() const = 0;
		virtual void zero() = 0;

	public:
		static void Sum_of_rows(Matrix &output, const Matrix &input) {
			assert(typeid(output) == typeid(input));
			assert(output.getRowLength() == input.getRowLength());
			assert(output.getColumnLength() == 1);
			output.set_to_sum_of_rows(input);
		}

		static void Add(Matrix &output, const Matrix &lhs, const Matrix &rhs) {
			assert(typeid(output) == typeid(lhs));
			assert(typeid(output) == typeid(rhs));
			assert(lhs.getDimensions() == rhs.getDimensions());
			output.set_to_sum_of(lhs, rhs);
		}

		static void Minus_By(Matrix<T> &output, const Matrix<T> &input) {
			assert(typeid(output) == typeid(input));
			assert(output.getDimensions() == input.getDimensions());
			output.subtract_by(input);
		}

		static void Minus(Matrix<T> &output, const Matrix<T> &lhs, const Matrix<T> &rhs) {
			assert(typeid(output) == typeid(lhs));
			assert(typeid(output) == typeid(rhs));
			assert(lhs.getDimensions() == rhs.getDimensions());
			output.set_to_difference_of(lhs, rhs);
		}

		static void Multiply(Matrix<T> &output, const Matrix<T> &lhs, const Matrix<T> &rhs) {
			assert(typeid(output) == typeid(lhs));
			assert(typeid(output) == typeid(rhs));
			assert(lhs.getDimensions()[1] == rhs.getDimensions()[0]);
			assert(output.getDimensions()[0] == lhs.getDimensions()[0]);
			assert(output.getDimensions()[1] == rhs.getDimensions()[1]);
			output.set_to_product_of(lhs, rhs);
		}

		static void Per_Element_Sigmoid(Matrix<T> &output, const Matrix<T> &input) {
			assert(typeid(output) == typeid(input));
			assert(output.getElems().size() == input.getElems().size());
			output.per_Element_Sigmoid(input);
		}

		static void Per_Element_Sigmoid_Prime(Matrix<T> &output, const Matrix<T> &sigmoid_value) {
			assert(typeid(output) == typeid(sigmoid_value));
			assert(output.getElems().size() == sigmoid_value.getElems().size());
			output.per_Element_Sigmoid_Prime(sigmoid_value);
		}

		static void Per_Element_Tanh(Matrix<T> &output, const Matrix<T> &input) {
			assert(typeid(output) == typeid(input));
			assert(output.getElems().size() == input.getElems().size());
			output.per_Element_Tanh(input);
		}

		static void Per_Element_Tanh_Prime(Matrix<T> &output, const Matrix<T> &tanh_value) {
			assert(typeid(output) == typeid(tanh_value));
			assert(output.getElems().size() == tanh_value.getElems().size());
			output.per_Element_Tanh_Prime(tanh_value);
		}

		/*
		1. For each column i in multiplicand, multiply by multipliers(0,i)
		2. Assign output to the transpose of the result of (1)
		*/
		static void Per_Column_Multiply_AndThen_Transpose(Matrix<T> &output, const Matrix<T> &multipliers, const Matrix<T> &multiplicand) {
			assert(typeid(output) == typeid(multipliers));
			assert(typeid(output) == typeid(multiplicand));
			assert(multipliers.getColumnLength() == 1);
			assert(output.getRowLength() == multiplicand.getColumnLength());
			assert(output.getColumnLength() == multiplicand.getRowLength());
			output.per_Column_Multiply_AndThen_Transpose(multipliers, multiplicand);
		}

		/*
		1. For each column i in multiplicand, multiply by multipliers(0,i)
		2. Multiply each value in (1) by scale
		3. Assign output to the value in (2)
		*/
		static void Per_Column_Multiply_AndThen_Scale(Matrix<T> &output, const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) {
			assert(typeid(output) == typeid(multipliers));
			assert(typeid(output) == typeid(multiplicand));
			assert(multipliers.getColumnLength() == 1);
			assert(output.getColumnLength() == multiplicand.getColumnLength());
			assert(output.getRowLength() == multiplicand.getRowLength());
			output.per_Column_Multiply_AndThen_Scale(multipliers, multiplicand, scale);
		}

		/*
		1. For each row i in multiplicand, multiply by multipliers(0,i)
		2. Assign output to the value in (2)
		*/
		static void Per_Row_Multiply(Matrix<T> &output, const Matrix<T> &multipliers, const Matrix<T> &multiplicand) {
			assert(typeid(output) == typeid(multipliers));
			assert(typeid(output) == typeid(multiplicand));
			assert(multipliers.getColumnLength() == 1);
			assert(output.getColumnLength() == multiplicand.getColumnLength());
			assert(output.getRowLength() == multiplicand.getRowLength());
			output.per_Row_Multiply(multipliers, multiplicand);
		}

		static void Row_Vectors_Per_Element_Multiply_AndThen_Scale(Matrix<T> &output, const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) {
			assert(typeid(output) == typeid(row_vector_1));
			assert(typeid(output) == typeid(row_vector_2));
			assert(row_vector_1.getColumnLength() == 1);
			assert(row_vector_2.getColumnLength() == 1);
			assert(output.getColumnLength() == 1);
			assert(output.getRowLength() == row_vector_1.getRowLength());
			assert(output.getRowLength() == row_vector_2.getRowLength());
			output.row_Vectors_Per_Element_Multiply_AndThen_Scale(row_vector_1, row_vector_2, scale);
		}

		static void Copy(Matrix<T> &output, const Matrix<T> &input) {
			assert(typeid(output) == typeid(input));
			assert(output.getDimensions() == input.getDimensions());
			output.copy(input);
		}

		static void Outer_product(Matrix<T> &output, const Matrix<T> &input1, const Matrix<T> &input2) {
			assert(typeid(output) == typeid(input1));
			assert(typeid(output) == typeid(input2));
			assert(input1.getColumnLength() == 1);
			assert(input2.getColumnLength() == 1);
			output.outer_product(input1, input2);
		}

		static void Copy_from_vector(Matrix<T> &output, const std::vector<T> &input) {
			assert(output.getRowLength()*output.getColumnLength() == input.size());
			output.copy_from_vector(input);
		}
	};

	template<typename T>
	bool operator==(const Matrix<T> &lhs, const Matrix<T> &rhs) {
		double tolerance = 0.0000001;
		bool retval = true;

		if (lhs.getDimensions() != rhs.getDimensions()) {
			retval = false;
		}
		else {
			auto lhs_elems = lhs.getElems();
			auto rhs_elems = rhs.getElems();

			for (int i = 0; i < lhs_elems.size(); i++) {
				if (abs(rhs_elems[i] - lhs_elems[i]) > tolerance) {
					retval = false;
				}
			}
		}
		return retval;
	};
}