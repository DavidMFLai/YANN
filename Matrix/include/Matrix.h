#pragma once

#include <array>
#include <cassert>

template <typename T>
class Matrix {
public:
	//Getting dimensions
	std::array<size_t, 2> getDimensions() const {
		return this->matrixAccessProperties.dimensions;
	}

	//Getting No. of rows
	size_t getRowCount() const {
		return this->matrixAccessProperties.dimensions[0];
	}
	//Getting No. of rows
	size_t getColumnCount() const {
		return this->matrixAccessProperties.dimensions[1];
	}

protected:
	struct MatrixAccessProperties {
		void setDimensions(size_t rowCount, size_t columnCount) {
			this->dimensions = { rowCount, columnCount };
		}

		size_t operator()(size_t i, size_t j) const {
			assert(i < this->dimensions[0]);
			assert(j < this->dimensions[1]);
			return i*this->dimensions[1] + j;
		}

		std::array<size_t, 2> dimensions;
	};

protected:
	MatrixAccessProperties matrixAccessProperties;


private:
	virtual void subtract_andThen_assign(const Matrix<T> &) = 0;
	virtual void sum_of_rows(const Matrix<T> &input) = 0;
	virtual void add(const Matrix<T> &lhs, const Matrix<T> &rhs) = 0;
	virtual void minus(const Matrix<T> &lhs, const Matrix<T> &rhs) = 0;
	virtual void multiply(const Matrix<T> &lhs, const Matrix<T> &rhs) = 0;
	virtual void per_Element_Sigmoid(const Matrix<T> &input) = 0;
	virtual void per_Element_Sigmoid_Prime(const Matrix<T> &sigmoid_value) = 0;
	virtual void per_Element_Tanh(const Matrix<T> &input) = 0;
	virtual void per_Element_Tanh_Prime(const Matrix<T> &tanh_value) = 0;



	virtual void per_Column_Multiply_AndThen_Transpose(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) = 0;
	virtual void per_Column_Multiply_AndThen_Scale(const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) = 0;
	virtual void per_Row_Multiply(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) = 0;
	virtual void row_Vectors_Per_Element_Multiply_AndThen_Scale(const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) = 0;

public:
	static void subtract_andThen_assign(Matrix<T> &output, const Matrix<T> &input) {
		assert(typeid(output) == typeid(input));
		assert(output.getDimensions() == input.getDimensions());
		output.subtract_andThen_assign(input);
	}

	static void Sum_of_rows(Matrix &output, const Matrix &input) {
		assert(typeid(output) == typeid(input));
		assert(output.getColumnCount() == input.getColumnCount());
		assert(output.getRowCount() == 1);
		output.sum_of_rows(input);
	}

	static void Add(Matrix &output, const Matrix &lhs, const Matrix &rhs) {
		assert(typeid(output) == typeid(lhs));
		assert(typeid(output) == typeid(rhs));
		assert(lhs.getDimensions() == rhs.getDimensions());
		output.add(lhs, rhs);
	}

	static void Minus(Matrix<T> &output, const Matrix<T> &lhs, const Matrix<T> &rhs) {
		assert(typeid(output) == typeid(lhs));
		assert(typeid(output) == typeid(rhs));
		assert(lhs.getDimensions() == rhs.getDimensions());
		output.minus(lhs, rhs);
	}

	static void Multiply(Matrix<T> &output, const Matrix<T> &lhs, const Matrix<T> &rhs) {
		assert(typeid(output) == typeid(lhs));
		assert(typeid(output) == typeid(rhs));
		assert(lhs.getDimensions()[1] == rhs.getDimensions()[0]);
		assert(output.getDimensions()[0] == lhs.getDimensions()[0]);
		assert(output.getDimensions()[1] == rhs.getDimensions()[1]);
		output.multiply(lhs, rhs);
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
		assert(multipliers.getRowCount() == 1);
		assert(output.getColumnCount() == multiplicand.getRowCount());
		assert(output.getRowCount() == multiplicand.getColumnCount());
		output.per_Column_Multiply_AndThen_Transpose(multipliers, multiplicand);

		//for (size_t i = 0; i < output.getRowCount(); i++) {
		//	for (size_t j = 0; j < output.getColumnCount(); j++) {
		//		output(i, j) = multipliers(0, i) * multiplicand(j, i);
		//	}
		//}
	}

	/*
	1. For each column i in multiplicand, multiply by multipliers(0,i)
	2. Multiply each value in (1) by scale
	3. Assign output to the value in (2)
	*/
	static void Per_Column_Multiply_AndThen_Scale(Matrix<T> &output, const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) {
		assert(typeid(output) == typeid(multipliers));
		assert(typeid(output) == typeid(multiplicand));
		assert(multipliers.getRowCount() == 1);
		assert(output.getRowCount() == multiplicand.getRowCount());
		assert(output.getColumnCount() == multiplicand.getColumnCount());
		output.per_Column_Multiply_AndThen_Scale(multipliers, multiplicand, scale);

		//for (size_t i = 0; i < output.getrowcount(); i++) {
		//	for (size_t j = 0; j < output.getcolumncount(); j++) {
		//		output(i, j) = multipliers(0, j) * multiplicand(i, j) * scale;
		//	}
		//}
	}

	/*
	1. For each row i in multiplicand, multiply by multipliers(0,i)
	2. Assign output to the value in (2)
	*/
	static void Per_Row_Multiply(Matrix<T> &output, const Matrix<T> &multipliers, const Matrix<T> &multiplicand) {
		assert(typeid(output) == typeid(multipliers));
		assert(typeid(output) == typeid(multiplicand));
		assert(multipliers.getRowCount() == 1);
		assert(output.getRowCount() == multiplicand.getRowCount());
		assert(output.getColumnCount() == multiplicand.getColumnCount());
		output.per_Row_Multiply(multipliers, multiplicand);

		//for (size_t i = 0; i < output.getRowCount(); i++) {
		//	for (size_t j = 0; j < output.getColumnCount(); j++) {
		//		output(i, j) = multipliers(0, i) * multiplicand(i, j);
		//	}
		//}
	}

	static void Row_Vectors_Per_Element_Multiply_AndThen_Scale(Matrix<T> &output, const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) {
		assert(typeid(output) == typeid(row_vector_1));
		assert(typeid(output) == typeid(row_vector_2));
		assert(row_vector_1.getRowCount() == 1);
		assert(row_vector_2.getRowCount() == 1);
		assert(output.getRowCount() == 1);
		assert(output.getColumnCount() == row_vector_1.getColumnCount());
		assert(output.getColumnCount() == row_vector_2.getColumnCount());
		output.row_Vectors_Per_Element_Multiply_AndThen_Scale(row_vector_1, row_vector_2, scale);


		//for (size_t i = 0; i < output.getColumnCount(); i++) {
		//	output(0, i) = row_vector_1(0, i) * row_vector_2(0, i) * scale;
		//}
	}
};