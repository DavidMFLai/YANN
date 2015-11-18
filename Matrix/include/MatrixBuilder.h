#pragma once

#include <initializer_list>
#include "Matrix.h"
#include <memory>

template <typename T>
class MatrixBuilder {
public:
	virtual std::unique_ptr<Matrix<T>> create(size_t rowCount, size_t columnCount) = 0;
	virtual std::unique_ptr<Matrix<T>> create(std::initializer_list<std::initializer_list<T>> t) = 0;
	virtual std::unique_ptr<Matrix<T>> create(const std::vector<std::vector<T>> &t) = 0;
	virtual std::unique_ptr<Matrix<T>> create(const std::vector<T> &t) = 0; //creates row matrix
};