#pragma once
#include "ReferenceMatrix.h"
#include "MatrixBuilder.h"
#include <vector>

template <typename T>
class ReferenceMatrixBuilder : public MatrixBuilder<T> {
public:
	ReferenceMatrixBuilder<T>()
	{}
	
	ReferenceMatrixBuilder<T> &setDimensions(size_t rowCount, size_t columnCount) override {
		data.resize(rowCount);
		for (auto &data_row : data) {
			data_row.resize(columnCount);
		}
		return *this;
	};

	ReferenceMatrixBuilder<T> &setValues(std::initializer_list<std::initializer_list<T>> lists) override {
		for (auto list : lists) {
			data.push_back(std::vector<T>{list});
		}
		return *this;
	};

	ReferenceMatrixBuilder<T> &setValues(const std::vector<std::vector<T>> &v) override {
		data = v;
		return *this;
	};

	std::unique_ptr<Matrix<T>> build() override {
		std::unique_ptr<Matrix<T>> retval{ new ReferenceMatrix<T>{ data } };
		return retval;
	}

private:
	std::vector<std::vector<T>> data;
};