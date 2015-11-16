#pragma once

#include <array>
#include <cassert>

class Matrix {
public:
	//Getting dimensions
	std::array<size_t, 2> getDimensions() const {
		return this->matrixAccessProperties.dimensions;
	}

	//Getting No. of rows
	size_t getRowCount() const {
		return this->matrixAccessProperties.dimensions[0];
	}
	//Getting No. of rows
	size_t getColumnCount() const {
		return this->matrixAccessProperties.dimensions[1];
	}

protected:
	struct MatrixAccessProperties {
		void setDimensions(size_t rowCount, size_t columnCount) {
			this->dimensions = { rowCount, columnCount };
		}

		size_t operator()(size_t i, size_t j) const {
			assert(i < this->dimensions[0]);
			assert(j < this->dimensions[1]);
			return i*this->dimensions[1] + j;
		}

		std::array<size_t, 2> dimensions;
	};

protected:
	MatrixAccessProperties matrixAccessProperties;
};