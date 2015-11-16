#pragma once

template <typename T>
class ReferenceMatrixBuilder : public MatrixBuilder {
public:
	ReferenceMatrixBuilder &setDimensions(size_t rowCount, size_t columnCount) override {
		
	};

	ReferenceMatrixBuilder &setValues(std::initializer_list<std::initializer_list<T>> t) override {

	};
};