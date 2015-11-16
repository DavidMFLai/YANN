#pragma once
#include <vector>
#include <array>
#include <cassert>
#include <tuple>
#include <cmath>

template<typename T>
class Matrix {
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

public:
	//defaulted constructors and destructors
	Matrix() = default;
 	Matrix(Matrix &&) = default;
	Matrix(const Matrix &) = default;
	Matrix &operator=(Matrix &&) = default;
	Matrix &operator=(const Matrix &) = default;
	~Matrix() = default;

	//Constructor by row and column size
	Matrix(size_t rowCount, size_t columnCount)
		:elems(rowCount*columnCount) {
		this->matrixAccessProperties.setDimensions(rowCount, columnCount);
	}

	//Constructor by initialization list
	Matrix(std::initializer_list<std::initializer_list<T>> lists) {
		this->matrixAccessProperties.setDimensions(lists.size(), lists.begin()->size());
		for (std::initializer_list<T> list : lists) {
			this->elems.insert(elems.end(), list);
		}
	}

	//Constructor by vector (only for a 1-by-n Matrix)
	Matrix(const std::vector<T> &list) {
		this->matrixAccessProperties.setDimensions(1, list.size());
		this->elems = list;
	}

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

	//Getting element(i,j)
	T& operator()(size_t i, size_t j) {
		return this->elems[matrixAccessProperties(i, j)];
	}

	//Getting element(i,j), const version
	const T& operator()(size_t i, size_t j) const {
		return this->elems[matrixAccessProperties(i, j)];
	}

	Matrix &operator-=(const Matrix<T> &rhs) {
		assert(this->getDimensions() == rhs.getDimensions());
		for (size_t idx = 0; idx < elems.size(); ++idx) {
			this->elems[idx] -= rhs.elems[idx];
		}
		return *this;
	}

	static void Sum_of_rows(Matrix &output, const Matrix &input) {
		assert(output.getColumnCount() == input.getColumnCount());
		for (size_t i = 0; i < input.getColumnCount(); ++i) {
			output(0, i) = 0;
			for (size_t j = 0; j < input.getRowCount(); ++j)
				output(0, i) += input(j, i);
		}
	}

	static void Add(Matrix &output, const Matrix &lhs, const Matrix &rhs) {
		//output = lhs + rhs;
		assert(lhs.getDimensions() == rhs.getDimensions());
		for (size_t idx = 0; idx < lhs.elems.size(); ++idx) {
			output.elems[idx] = lhs.elems[idx] + rhs.elems[idx];
		}
	}

	static void Minus(Matrix &output, const Matrix &lhs, const Matrix &rhs) {
		//output = lhs - rhs;
		assert(lhs.getDimensions() == rhs.getDimensions());
		for (size_t idx = 0; idx < lhs.elems.size(); ++idx) {
			output.elems[idx] = lhs.elems[idx] - rhs.elems[idx];
		}
	}

	static void Multiply(Matrix &output, const Matrix &lhs, const Matrix &rhs) {
		//output = lhs * rhs;
		assert(lhs.getDimensions()[1] == rhs.getDimensions()[0]);
		assert(output.getDimensions()[0] == lhs.getDimensions()[0]);
		assert(output.getDimensions()[1] == rhs.getDimensions()[1]);
		for (size_t m = 0; m < lhs.getDimensions()[0]; m++) {
			for (size_t n = 0; n < rhs.getDimensions()[1]; n++) {
				output(m, n) = 0;
				for (size_t p = 0; p < lhs.getDimensions()[1]; p++) {
					output(m, n) += lhs(m, p)*rhs(p, n);
				}
			}
		}
	}

	static void Per_Element_Sigmoid(Matrix &output, const Matrix &input) {
		assert(output.getElems().size() == input.getElems().size());
		for (size_t idx = 0; idx < output.getElems().size(); idx++) {
			output.getElems()[idx] = 1 / (1 + std::exp(-input.getElems()[idx]));
		}
	}

	static void Per_Element_Sigmoid_Prime(Matrix &output, const Matrix &sigmoid_value) {
		assert(output.getElems().size() == sigmoid_value.getElems().size());
		for (size_t idx = 0; idx < output.getElems().size(); idx++) {
			output.getElems()[idx] = sigmoid_value.getElems()[idx] * (1 - sigmoid_value.getElems()[idx]);
		}
	}

	static void Per_Element_Tanh(Matrix &output, const Matrix &input) {
		assert(output.getElems().size() == input.getElems().size());
		for (size_t idx = 0; idx < output.getElems().size(); idx++) {
			output.getElems()[idx] = std::tanh(input.getElems()[idx]);
		}
	}

	static void Per_Element_Tanh_Prime(Matrix &output, const Matrix &tanh_value) {
		assert(output.getElems().size() == tanh_value.getElems().size());
		for (size_t idx = 0; idx < output.getElems().size(); idx++) {
			output.getElems()[idx] = 1 - (tanh_value.getElems()[idx] * tanh_value.getElems()[idx]);
		}
	}

	/*
	1. For each column i in multiplicand, multiply by multipliers(0,i)
	2. Assign output to the transpose of the result of (1)
	*/
	static void Per_Column_Multiply_AndThen_Transpose(Matrix &output, const Matrix &multipliers, const Matrix &multiplicand) {
		assert(multipliers.getRowCount() == 1);
		assert(output.getColumnCount() == multiplicand.getRowCount());
		assert(output.getRowCount() == multiplicand.getColumnCount());

		for (size_t i = 0; i < output.getRowCount(); i++) {
			for (size_t j = 0; j < output.getColumnCount(); j++) {
				output(i, j) = multipliers(0, i) * multiplicand(j, i);
			}
		}
	}

	/*
	1. For each column i in multiplicand, multiply by multipliers(0,i)
	2. Multiply each value in (1) by scale
	3. Assign output to the value in (2)
	*/
	static void Per_Column_Multiply_AndThen_Scale(Matrix &output, const Matrix &multipliers, const Matrix &multiplicand, T scale) {
		assert(multipliers.getRowCount() == 1);
		assert(output.getColumnCount() == multiplicand.getRowCount());
		assert(output.getRowCount() == multiplicand.getColumnCount());

		for (size_t i = 0; i < output.getRowCount(); i++) {
			for (size_t j = 0; j < output.getColumnCount(); j++) {
				output(i, j) = multipliers(0, j) * multiplicand(i, j) * scale;
			}
		}
	}

	/*
	1. For each row i in multiplicand, multiply by multipliers(0,i)
	2. Assign output to the value in (2)
	*/
	static void Per_Row_Multiply(Matrix &output, const Matrix &multipliers, const Matrix &multiplicand) {
		assert(multipliers.getRowCount() == 1);
		assert(output.getColumnCount() == multiplicand.getRowCount());
		assert(output.getRowCount() == multiplicand.getColumnCount());

		for (size_t i = 0; i < output.getRowCount(); i++) {
			for (size_t j = 0; j < output.getColumnCount(); j++) {
				output(i, j) = multipliers(0, i) * multiplicand(i, j);
			}
		}
	}

	static void Row_Vectors_Per_Element_Multiply_AndThen_Scale(Matrix &output, const Matrix &row_vector_1, const Matrix &row_vector_2, T scale) {
		assert(row_vector_1.getRowCount() == 1);
		assert(row_vector_2.getRowCount() == 1);
		assert(output.getRowCount() == 1);
		assert(output.getColumnCount() == row_vector_1.getColumnCount());
		assert(output.getColumnCount() == row_vector_2.getColumnCount());

		for (size_t i = 0; i < output.getColumnCount(); i++) {
			output(0, i) = row_vector_1(0, i) * row_vector_2(0, i) * scale;
		}
	}

	static void copy(Matrix &output, const Matrix &input) {
		assert(output.getDimensions() == input.getDimensions());

		for (size_t i = 0; i < output.getRowCount(); i++) {
			for (size_t j = 0; j < output.getColumnCount(); j++) {
				output(i, j) = input(i, j);
			}
		}
	}

	static void Outer_product(Matrix &output, const Matrix &input1, const Matrix &input2) {
		assert(input1.getRowCount() == 1);
		assert(input2.getRowCount() == 1);

		for (size_t i = 0; i < output.getRowCount(); i++) {
			for (size_t j = 0; j < output.getColumnCount(); j++) {
				output(i, j) = input1(0, i) * input2(0, j);
			}
		}
	}

	static void Copy_from_vector(Matrix &output, const std::vector<T> &input) {
		assert(output.getColumnCount()*output.getRowCount() == input.size());
		std::copy(input.begin(), input.end(), output.elems.begin());
	}

	void zero() {
		for (auto &elem : elems) {
			elem = 0;
		}
	}

public:
	const std::vector<T> &getElems() const{
		return elems;
	}

	std::vector<T> &getElems() {
		return elems;
	}

private:
	MatrixAccessProperties matrixAccessProperties;
	std::vector<T> elems;
};


template<typename T>
bool operator==(const Matrix<T> &lhs, const Matrix<T> &rhs) {
	double tolerance = 0.0000001;
	
	if (lhs.getDimensions() != rhs.getDimensions()) {
		return false;
	}

	for (size_t i = 0; i < lhs.getDimensions()[0]; i++)
		for (size_t j = 0; j < lhs.getDimensions()[1]; j++)
			if (abs(lhs(i, j) - rhs(i, j)) > tolerance)
				return false;

	return true;
}