#pragma once

#include "Matrix.h"
#include "OpenCLMatrixBuilder.h"

template<typename T>
class OpenCLMatrix : public Matrix<T> {
private:
	friend class OpenCLMatrixBuilder<T>;

	//Constructor by vector of vectors
	OpenCLMatrix(std::vector<std::vector<T>> lists, const map<string, cl::Kernel> &kernels)
		: kernels{kernels}{
		this->matrixAccessProperties.setDimensions(lists.size(), lists.begin()->size());
		//todo: Add contents..
	}

	//deleted default, copy, move, copy assignment, and move assignment
	OpenCLMatrix() = delete;
	OpenCLMatrix(OpenCLMatrix &&) = delete;
	OpenCLMatrix(const OpenCLMatrix &) = delete;
	OpenCLMatrix &operator=(OpenCLMatrix &&) = delete;
	OpenCLMatrix &operator=(const OpenCLMatrix &) = delete;

private:
	virtual bool is_equal(const Matrix<T> &) const {
		return false; //todo: implement
	};

	virtual void subtract_andThen_assign(const Matrix<T> &) {
		return; //todo: implement
	}
	virtual void sum_of_rows(const Matrix<T> &input) {
		return;
	}
	virtual void add(const Matrix<T> &lhs, const Matrix<T> &rhs) {
		return;
	}
	virtual void minus(const Matrix<T> &lhs, const Matrix<T> &rhs) {
		return;
	}
	virtual void multiply(const Matrix<T> &lhs, const Matrix<T> &rhs) {
		return;
	}
	virtual void per_Element_Sigmoid(const Matrix<T> &input) {
		return;
	}
	virtual void per_Element_Sigmoid_Prime(const Matrix<T> &sigmoid_value) {
		return;
	}
	virtual void per_Element_Tanh(const Matrix<T> &input) {
		return;
	}
	virtual void per_Element_Tanh_Prime(const Matrix<T> &tanh_value) {
		return;
	}
	virtual void per_Column_Multiply_AndThen_Transpose(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) {
		return;
	}
	virtual void per_Column_Multiply_AndThen_Scale(const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) {
		return;
	}
	virtual void per_Row_Multiply(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) {
		return;
	}
	virtual void row_Vectors_Per_Element_Multiply_AndThen_Scale(const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) {
		return;
	}
	virtual void copy(const Matrix<T> &input) {
		return;
	}
	virtual void outer_product(const Matrix<T> &input1, const Matrix<T> &input2) {
		return;
	}
	virtual void copy_from_vector(const std::vector<T> &input) {
		return;
	}

public:

	T temp;
	virtual T &at(size_t i, size_t j) {
		return temp;
	}
	virtual const T &at(size_t i, size_t j) const {
		return T{};
	}
	virtual std::vector<T> &getElems() {
		return std::vector<T>{};
	}
	virtual const std::vector<T> &getElems() const {
		return std::vector<T>{};
	}
	virtual void zero() {
		return;
	}
	
private:
	const unordered_map<string, cl::Kernel> &kernels;
};