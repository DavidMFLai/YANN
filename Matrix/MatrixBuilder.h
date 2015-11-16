#pragma once

#include <initializer_list>
#include "Matrix.h"

template <typename T>
class MatrixBuilder {
public:
	virtual MatrixBuilder &setDimensions(size_t rowCount, size_t columnCount) = 0;
	virtual MatrixBuilder &setValues(std::initializer_list<std::initializer_list<T>> t) = 0;
	virtual Matrix &build() = 0;
};