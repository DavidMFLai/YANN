#pragma once

#include "Matrix.h"

#include <vector>
#include <array>
#include <cassert>
#include <tuple>
#include <cmath>

template<typename T>
class ReferenceMatrix : public Matrix<T> {
public:
	//defaulted constructors and destructors
	ReferenceMatrix() 
		: ReferenceMatrix(0, 0) 
	{};

 	ReferenceMatrix(ReferenceMatrix &&) = default;
	ReferenceMatrix(const ReferenceMatrix &) = default;
	ReferenceMatrix &operator=(ReferenceMatrix &&) = default;
	ReferenceMatrix &operator=(const ReferenceMatrix &) = default;
	~ReferenceMatrix() = default;

	//Constructor by row and column size
	ReferenceMatrix(size_t rowCount, size_t columnCount)
		:elems(rowCount*columnCount) {
		this->matrixAccessProperties.setDimensions(rowCount, columnCount);
	}

	//Constructor by initialization list
	ReferenceMatrix(std::initializer_list<std::initializer_list<T>> lists) {
		this->matrixAccessProperties.setDimensions(lists.size(), lists.begin()->size());
		for (std::initializer_list<T> list : lists) {
			this->elems.insert(elems.end(), list);
		}
	}

	//Constructor by vector of vectors
	ReferenceMatrix(std::vector<std::vector<T>> lists) {
		this->matrixAccessProperties.setDimensions(lists.size(), lists.begin()->size());
		for (std::vector<T> list : lists) {
			this->elems.insert(elems.end(), list.begin(), list.end());
		}
	}

	//Constructor by vector (only for a 1-by-n Matrix)
	ReferenceMatrix(const std::vector<T> &list) {
		this->matrixAccessProperties.setDimensions(1, list.size());
		this->elems = list;
	}

	//Getting element(i,j)
	T& operator()(size_t i, size_t j) {
		return this->elems[matrixAccessProperties(i, j)];
	}

	//Getting element(i,j), const version
	const T& operator()(size_t i, size_t j) const {
		return this->elems[matrixAccessProperties(i, j)];
	}

	//Getting element(i,j)
	T& at(size_t i, size_t j) {
		return this->elems[matrixAccessProperties(i, j)];
	}

	//Getting element(i,j), function form, const version
	const T& at(size_t i, size_t j) const {
		return this->elems[matrixAccessProperties(i, j)];
	}

private:
	void subtract_andThen_assign(const Matrix<T> &input) override {
		const ReferenceMatrix<T> &input_as_reference_matrix = static_cast<const ReferenceMatrix<T> &>(input);
		for (size_t idx = 0; idx < this->getElems().size(); ++idx) {
			this->elems[idx] -= input_as_reference_matrix.elems[idx];
		}
	}

	void sum_of_rows(const Matrix<T> &input) override {
		const ReferenceMatrix<T> &input_as_reference_matrix = static_cast<const ReferenceMatrix<T> &>(input);
		for (size_t i = 0; i < input_as_reference_matrix.getColumnCount(); ++i) {
			this->at(0, i) = 0;
			for (size_t j = 0; j < input_as_reference_matrix.getRowCount(); ++j)
				this->at(0, i) += input_as_reference_matrix(j, i);
		}
	}

	void add(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
		const ReferenceMatrix<T> &lhs_rm = static_cast<const ReferenceMatrix<T> &>(lhs);
		const ReferenceMatrix<T> &rhs_rm = static_cast<const ReferenceMatrix<T> &>(rhs);
		
		for (size_t idx = 0; idx < lhs_rm.elems.size(); ++idx) {
			this->elems[idx] = lhs_rm.elems[idx] + rhs_rm.elems[idx];
		}
	}

	void minus(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
		const ReferenceMatrix<T> &lhs_rm = static_cast<const ReferenceMatrix<T> &>(lhs);
		const ReferenceMatrix<T> &rhs_rm = static_cast<const ReferenceMatrix<T> &>(rhs);
		for (size_t idx = 0; idx < lhs_rm.elems.size(); ++idx) {
			this->elems[idx] = lhs_rm.elems[idx] - rhs_rm.elems[idx];
		}
	}

	void multiply(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
		const ReferenceMatrix<T> &lhs_rm = static_cast<const ReferenceMatrix<T> &>(lhs);
		const ReferenceMatrix<T> &rhs_rm = static_cast<const ReferenceMatrix<T> &>(rhs);
		for (size_t m = 0; m < lhs_rm.getDimensions()[0]; m++) {
			for (size_t n = 0; n < rhs_rm.getDimensions()[1]; n++) {
				this->at(m, n) = 0;
				for (size_t p = 0; p < lhs_rm.getDimensions()[1]; p++) {
					this->at(m, n) += lhs_rm(m, p)*rhs_rm(p, n);
				}
			}
		}
	}

	void per_Element_Sigmoid(const Matrix<T> &input) override {
		const ReferenceMatrix<T> &input_rm = static_cast<const ReferenceMatrix<T> &>(input);
		for (size_t idx = 0; idx < this->getElems().size(); idx++) {
			this->getElems()[idx] = 1 / (1 + std::exp(-input_rm.getElems()[idx]));
		}
	}

	void per_Element_Sigmoid_Prime(const Matrix<T> &sigmoid_value) override {
		const ReferenceMatrix<T> &sigmoid_value_rm = static_cast<const ReferenceMatrix<T> &>(sigmoid_value);
		for (size_t idx = 0; idx < this->getElems().size(); idx++) {
			this->getElems()[idx] = sigmoid_value_rm.getElems()[idx] * (1 - sigmoid_value_rm.getElems()[idx]);
		}
	}

	void per_Element_Tanh(const Matrix<T> &input) override {
		const ReferenceMatrix<T> &input_rm = static_cast<const ReferenceMatrix<T> &>(input);
		for (size_t idx = 0; idx < this->getElems().size(); idx++) {
			this->getElems()[idx] = std::tanh(input_rm.getElems()[idx]);
		}
	}

	void per_Element_Tanh_Prime(const Matrix<T> &tanh_value) override {
		const ReferenceMatrix<T> &tanh_value_rm = static_cast<const ReferenceMatrix<T> &>(tanh_value);
		for (size_t idx = 0; idx < this->getElems().size(); idx++) {
			this->getElems()[idx] = 1 - (tanh_value_rm.getElems()[idx] * tanh_value_rm.getElems()[idx]);
		}
	};

	void per_Column_Multiply_AndThen_Transpose(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) {
		const ReferenceMatrix<T> &multipliers_rm = static_cast<const ReferenceMatrix<T> &>(multipliers);
		const ReferenceMatrix<T> &multiplicand_rm = static_cast<const ReferenceMatrix<T> &>(multiplicand);

		for (size_t i = 0; i < this->getRowCount(); i++) {
			for (size_t j = 0; j < this->getColumnCount(); j++) {
				this->at(i, j) = multipliers_rm.at(0, i) * multiplicand_rm.at(j, i);
			}
		}
	}

	void per_Column_Multiply_AndThen_Scale(const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) {
		const ReferenceMatrix<T> &multipliers_rm = static_cast<const ReferenceMatrix<T> &>(multipliers);
		const ReferenceMatrix<T> &multiplicand_rm = static_cast<const ReferenceMatrix<T> &>(multiplicand);
		for (size_t i = 0; i < this->getRowCount(); i++) {
			for (size_t j = 0; j < this->getColumnCount(); j++) {
				this->at(i, j) = multipliers_rm(0, j) * multiplicand_rm(i, j) * scale;
			}
		}
	}

	void per_Row_Multiply(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) {
		const ReferenceMatrix<T> &multipliers_rm = static_cast<const ReferenceMatrix<T> &>(multipliers);
		const ReferenceMatrix<T> &multiplicand_rm = static_cast<const ReferenceMatrix<T> &>(multiplicand);
		for (size_t i = 0; i < this->getRowCount(); i++) {
			for (size_t j = 0; j < this->getColumnCount(); j++) {
				this->at(i, j) = multipliers_rm(0, i) * multiplicand_rm(i, j);
			}
		}
	}

	void row_Vectors_Per_Element_Multiply_AndThen_Scale(const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) {
		const ReferenceMatrix<T> &row_vector_1_rm = static_cast<const ReferenceMatrix<T> &>(row_vector_1);
		const ReferenceMatrix<T> &row_vector_2_rm = static_cast<const ReferenceMatrix<T> &>(row_vector_2);

		for (size_t i = 0; i < this->getColumnCount(); i++) {
			this->at(0, i) = row_vector_1_rm(0, i) * row_vector_2_rm(0, i) * scale;
		}
	}

	void copy(const Matrix<T> &input) override {
		const ReferenceMatrix<T> &input_rm = static_cast<const ReferenceMatrix<T> &>(input);
		for (size_t i = 0; i < this->getRowCount(); i++) {
			for (size_t j = 0; j < this->getColumnCount(); j++) {
				this->at(i, j) = input_rm(i, j);
			}
		}
	}

	void outer_product(const Matrix<T> &input1, const Matrix<T> &input2) override {
		const ReferenceMatrix<T> &input1_rm = static_cast<const ReferenceMatrix<T> &>(input1);
		const ReferenceMatrix<T> &input2_rm = static_cast<const ReferenceMatrix<T> &>(input2);

		for (size_t i = 0; i < this->getRowCount(); i++) {
			for (size_t j = 0; j < this->getColumnCount(); j++) {
				this->at(i, j) = input1_rm(0, i) * input2_rm(0, j);
			}
		}
	}

	void copy_from_vector(const std::vector<T> &input) override {
		std::copy(input.begin(), input.end(), this->elems.begin());
	}

public:
	const std::vector<T> &getElems() const{
		return elems;
	}

	std::vector<T> &getElems() {
		return elems;
	}

private:
	std::vector<T> elems;
};


template<typename T>
bool operator==(const ReferenceMatrix<T> &lhs, const ReferenceMatrix<T> &rhs) {
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