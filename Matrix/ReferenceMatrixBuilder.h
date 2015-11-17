#pragma once

#include "MatrixBuilder.h"
#include "ReferenceMatrix.h"
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

	std::unique_ptr<Matrix> build() override {
		std::unique_ptr<Matrix> retval{ new ReferenceMatrix<T>{ data } };
		return retval;
	}

private:
	std::vector<std::vector<T>> data;
};