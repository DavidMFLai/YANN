#pragma once

#include "Matrix.h"
#include "ReferenceMatrixBuilder.h"

#include <vector>
#include <array>
#include <cassert>
#include <tuple>
#include <cmath>

template<typename T>
class ReferenceMatrix : public Matrix<T> {
private:
	friend class ReferenceMatrixBuilder<T>;

	//Constructor by vector of vectors
	ReferenceMatrix(std::vector<std::vector<T>> lists) {
		this->matrixAccessProperties.setDimensions(lists.size(), lists.begin()->size());
		for (const std::vector<T> &list : lists) {
			this->elems.insert(elems.end(), list.begin(), list.end());
		}
	}

	//deleted default, copy, move, copy assignment, and move assignment
	ReferenceMatrix() = delete;
 	ReferenceMatrix(ReferenceMatrix &&) = delete;
	ReferenceMatrix(const ReferenceMatrix &) = delete;
	ReferenceMatrix &operator=(ReferenceMatrix &&) = delete;
	ReferenceMatrix &operator=(const ReferenceMatrix &) = delete;

public:
	//Getting element(i,j)
	T& at(size_t i, size_t j) override{
		return this->elems[matrixAccessProperties(i, j)];
	}

	//Getting element(i,j), function form, const version
	const T& at(size_t i, size_t j) const override {
		return this->elems[matrixAccessProperties(i, j)];
	}

	void zero() override {
		std::fill(elems.begin(), elems.end(), 0);
	}

private:
	void subtract_by(const Matrix<T> &input) override {
		const ReferenceMatrix<T> &input_as_reference_matrix = static_cast<const ReferenceMatrix<T> &>(input);
		for (size_t idx = 0; idx < this->getElems().size(); ++idx) {
			this->elems[idx] -= input_as_reference_matrix.elems[idx];
		}
	}

	void set_to_sum_of_rows(const Matrix<T> &input) override {
		const ReferenceMatrix<T> &input_as_reference_matrix = static_cast<const ReferenceMatrix<T> &>(input);
		for (size_t i = 0; i < input_as_reference_matrix.getRowLength(); ++i) {
			this->at(0, i) = 0;
			for (size_t j = 0; j < input_as_reference_matrix.getColumnLength(); ++j)
				this->at(0, i) += input_as_reference_matrix.at(j, i);
		}
	}

	void set_to_sum_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
		const ReferenceMatrix<T> &lhs_rm = static_cast<const ReferenceMatrix<T> &>(lhs);
		const ReferenceMatrix<T> &rhs_rm = static_cast<const ReferenceMatrix<T> &>(rhs);
		
		for (size_t idx = 0; idx < lhs_rm.elems.size(); ++idx) {
			this->elems[idx] = lhs_rm.elems[idx] + rhs_rm.elems[idx];
		}
	}

	void set_to_difference_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
		const ReferenceMatrix<T> &lhs_rm = static_cast<const ReferenceMatrix<T> &>(lhs);
		const ReferenceMatrix<T> &rhs_rm = static_cast<const ReferenceMatrix<T> &>(rhs);
		for (size_t idx = 0; idx < lhs_rm.elems.size(); ++idx) {
			this->elems[idx] = lhs_rm.elems[idx] - rhs_rm.elems[idx];
		}
	}

	void set_to_product_of(const Matrix<T> &lhs, const Matrix<T> &rhs) override {
		const ReferenceMatrix<T> &lhs_rm = static_cast<const ReferenceMatrix<T> &>(lhs);
		const ReferenceMatrix<T> &rhs_rm = static_cast<const ReferenceMatrix<T> &>(rhs);
		for (size_t m = 0; m < lhs_rm.getDimensions()[0]; m++) {
			for (size_t n = 0; n < rhs_rm.getDimensions()[1]; n++) {
				this->at(m, n) = 0;
				for (size_t p = 0; p < lhs_rm.getDimensions()[1]; p++) {
					this->at(m, n) += lhs_rm.at(m, p)*rhs_rm.at(p, n);
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

		for (size_t i = 0; i < this->getColumnLength(); i++) {
			for (size_t j = 0; j < this->getRowLength(); j++) {
				this->at(i, j) = multipliers_rm.at(0, i) * multiplicand_rm.at(j, i);
			}
		}
	}

	void per_Column_Multiply_AndThen_Scale(const Matrix<T> &multipliers, const Matrix<T> &multiplicand, T scale) {
		const ReferenceMatrix<T> &multipliers_rm = static_cast<const ReferenceMatrix<T> &>(multipliers);
		const ReferenceMatrix<T> &multiplicand_rm = static_cast<const ReferenceMatrix<T> &>(multiplicand);
		for (size_t i = 0; i < this->getColumnLength(); i++) {
			for (size_t j = 0; j < this->getRowLength(); j++) {
				this->at(i, j) = multipliers_rm.at(0, j) * multiplicand_rm.at(i, j) * scale;
			}
		}
	}

	void per_Row_Multiply(const Matrix<T> &multipliers, const Matrix<T> &multiplicand) {
		const ReferenceMatrix<T> &multipliers_rm = static_cast<const ReferenceMatrix<T> &>(multipliers);
		const ReferenceMatrix<T> &multiplicand_rm = static_cast<const ReferenceMatrix<T> &>(multiplicand);
		for (size_t i = 0; i < this->getColumnLength(); i++) {
			for (size_t j = 0; j < this->getRowLength(); j++) {
				this->at(i, j) = multipliers_rm.at(0, i) * multiplicand_rm.at(i, j);
			}
		}
	}

	void row_Vectors_Per_Element_Multiply_AndThen_Scale(const Matrix<T> &row_vector_1, const Matrix<T> &row_vector_2, T scale) {
		const ReferenceMatrix<T> &row_vector_1_rm = static_cast<const ReferenceMatrix<T> &>(row_vector_1);
		const ReferenceMatrix<T> &row_vector_2_rm = static_cast<const ReferenceMatrix<T> &>(row_vector_2);

		for (size_t i = 0; i < this->getRowLength(); i++) {
			this->at(0, i) = row_vector_1_rm.at(0, i) * row_vector_2_rm.at(0, i) * scale;
		}
	}

	void copy(const Matrix<T> &input) override {
		const ReferenceMatrix<T> &input_rm = static_cast<const ReferenceMatrix<T> &>(input);
		for (size_t i = 0; i < this->getColumnLength(); i++) {
			for (size_t j = 0; j < this->getRowLength(); j++) {
				this->at(i, j) = input_rm.at(i, j);
			}
		}
	}

	void outer_product(const Matrix<T> &input1, const Matrix<T> &input2) override {
		const ReferenceMatrix<T> &input1_rm = static_cast<const ReferenceMatrix<T> &>(input1);
		const ReferenceMatrix<T> &input2_rm = static_cast<const ReferenceMatrix<T> &>(input2);

		for (size_t i = 0; i < this->getColumnLength(); i++) {
			for (size_t j = 0; j < this->getRowLength(); j++) {
				this->at(i, j) = input1_rm.at(0, i) * input2_rm.at(0, j);
			}
		}
	}

	void copy_from_vector(const std::vector<T> &input) override {
		std::copy(input.begin(), input.end(), this->elems.begin());
	}

public:
	const std::vector<T> &getElems() const override{
		return elems;
	}

	std::vector<T> &getElems() override {
		return elems;
	}

private:
	std::vector<T> elems;
};