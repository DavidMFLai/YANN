#pragma once

#include <initializer_list>
#include "Matrix.h"
#include <memory>

template <typename T>
class MatrixBuilder {
public:
	virtual MatrixBuilder<T> &setDimensions(size_t rowCount, size_t columnCount) = 0;
	virtual MatrixBuilder<T> &setValues(std::initializer_list<std::initializer_list<T>> t) = 0;
	virtual MatrixBuilder<T> &setValues(const std::vector<std::vector<T>> &t) = 0;
	virtual std::unique_ptr<Matrix<T>> build() = 0;
};