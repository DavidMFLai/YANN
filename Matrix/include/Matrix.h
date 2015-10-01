#pragma once
#include <vector>
#include <array>
#include <cassert>
#include <tuple>

template<typename T>
class Matrix_Ref {
	struct MatrixAccessProperties {
		void setDimensions(size_t rowCount, size_t columnCount, size_t stride) {
			dimensions = { rowCount, columnCount };
			this->stride = stride;
		}

		size_t operator()(size_t i, size_t j) const {
			return i*stride + j;
		}

		size_t stride;
		std::array<size_t, 2> dimensions;
	};

public:
	//defaulted constructors and destructors
	Matrix_Ref() = delete;
	Matrix_Ref(Matrix_Ref &&) = default;
	Matrix_Ref &operator=(Matrix_Ref &&) = default;
	Matrix_Ref(const Matrix_Ref &) = delete;
	Matrix_Ref &operator=(const Matrix_Ref &) = delete;
	~Matrix_Ref() = default;

	//Constructor by row and column size
	Matrix_Ref(size_t rowCount, size_t columnCount, size_t stride, T* dataStartingAddress)
		:elems(dataStartingAddress) {
		matrixAccessProperties.setDimensions(rowCount, columnCount, stride);
	};

	//Getting dimensions
	std::array<size_t, 2> getDimensions() const {
		return matrixAccessProperties.dimensions;
	}

	//Getting element(i,j)
	T& operator()(size_t i, size_t j) {
		return elems[matrixAccessProperties(i, j)];
	}

	//Getting element(i,j), const version
	const T& operator()(size_t i, size_t j) const {
		return elems[matrixAccessProperties(i, j)];
	}

private:
	MatrixAccessProperties matrixAccessProperties;
	T* elems;
};

template<typename T>
class Matrix {
	struct MatrixAccessProperties {
		void setDimensions(size_t rowCount, size_t columnCount) {
			dimensions = { rowCount, columnCount };
		}

		size_t operator()(size_t i, size_t j) const {
			return i*dimensions[1] + j;
		}

		std::array<size_t, 2> dimensions;
	};

public:
	//defaulted constructors and destructors
	Matrix(Matrix &&) = default;
	Matrix &operator=(Matrix &&) = default;
	Matrix(const Matrix &) = default;
	Matrix &operator=(const Matrix &) = default;
	~Matrix() = default;

	//default constructor
	Matrix()
		: Matrix(0, 0)
	{};

	//Constructor by row and column size
	Matrix(size_t rowCount, size_t columnCount)
		:elems(rowCount*columnCount) {
		matrixAccessProperties.setDimensions(rowCount, columnCount);
	};

	//Constructor by initialization list
	Matrix(std::initializer_list<std::initializer_list<T>> lists) {
		matrixAccessProperties.setDimensions(lists.size(), lists.begin()->size());
		for (std::initializer_list<T> list : lists) {
			elems.insert(elems.end(), list);
		}
	}

	//Constructor by vector (only for a n-by-1 Matrix)
	Matrix(const std::vector<T> &list) {
		matrixAccessProperties.setDimensions(list.size(), 1);
		elems = list;
	}

	//Constructor by sub matrix reference 
	Matrix(const Matrix_Ref<T> &matrix_ref)
		:Matrix(matrix_ref.getDimensions()[0], matrix_ref.getDimensions()[1]) {
		for (size_t i = 0; i < matrix_ref.getDimensions()[0]; i++)
			for (size_t j = 0; j < matrix_ref.getDimensions()[1]; j++)
				this->Matrix::operator()(i,j) = matrix_ref(i, j);
	}


	//Getting dimensions
	std::array<size_t, 2> getDimensions() const {
		return matrixAccessProperties.dimensions;
	}

	//Getting element(i,j)
	T& operator()(size_t i, size_t j) {
		return elems[matrixAccessProperties(i, j)];
	}

	//Getting element(i,j), const version
	const T& operator()(size_t i, size_t j) const {
		return elems[matrixAccessProperties(i, j)];
	}

	Matrix_Ref<T> getSubMatrixReference(std::tuple<size_t, size_t> row_range, std::tuple<size_t, size_t> column_range) {
		size_t starting_position = std::get<0>(row_range)*matrixAccessProperties.dimensions[1] + std::get<0>(column_range);
		T* starting_address = &elems[starting_position];
		Matrix_Ref<T> retval{ (std::get<1>(row_range) - std::get<0>(row_range) + 1 ),
			(std::get<1>(column_range) - std::get<0>(column_range) + 1),
			matrixAccessProperties.dimensions[1],
			starting_address
		};
		return retval;
	}
protected:
	std::vector<T> &getElems() {
		return elems;
	}

private:
	MatrixAccessProperties matrixAccessProperties;
	std::vector<T> elems;
};

// retval = lhs*rhs; lhs is a M*P matrix, rhs is a P*N matrix
template<typename T>  
Matrix<T> operator*(const Matrix<T> &lhs, const Matrix<T> &rhs) {
	assert(lhs.getDimensions()[1] == rhs.getDimensions()[0]);
	Matrix<T> retval{ lhs.getDimensions()[0], rhs.getDimensions()[1] };
	for (size_t m = 0; m < lhs.getDimensions()[0]; m++)
		for (size_t p = 0; p < lhs.getDimensions()[1]; p++)
			for (size_t n = 0; n < rhs.getDimensions()[1]; n++)
				retval(m, n) += lhs(m, p)*rhs(p, n);
	return retval;
};

template<typename T>
bool operator==(const Matrix<T> &lhs, const Matrix<T> &rhs) {
	if (lhs.getDimensions() != rhs.getDimensions()) {
		return false;
	}

	for (size_t i = 0; i < lhs.getDimensions()[0]; i++)
		for (size_t j = 0; j < lhs.getDimensions()[1]; j++)
			if (lhs(i, j) != rhs(i, j))
				return false;

	return true;
}