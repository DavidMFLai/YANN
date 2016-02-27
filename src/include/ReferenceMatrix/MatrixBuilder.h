#pragma once

#include "Matrix.h"
#include <initializer_list>
#include <memory>

namespace YANN{
	template <typename T>
	class MatrixBuilder {
	public:
		virtual std::unique_ptr<Matrix<T>> build(size_t rowCount, size_t columnCount) = 0;
		virtual std::unique_ptr<Matrix<T>> build(std::initializer_list<std::initializer_list<T>> t) = 0;
		virtual std::unique_ptr<Matrix<T>> build(const std::vector<std::vector<T>> &t) = 0;
		virtual std::unique_ptr<Matrix<T>> buildRowMatrix(const std::vector<T> &t) = 0; //creates row matrix
		virtual std::string getInfo() = 0;
		virtual ~MatrixBuilder<T>() = default;
	};
}