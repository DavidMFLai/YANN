#pragma once
#include <vector>
#include "ReferenceMatrix.h"
#include "MatrixBuilder.h"

namespace {
	template <typename T>
	class ReferenceMatrixBuilder : public MatrixBuilder<T> {
	public:
		ReferenceMatrixBuilder<T>()
		{}

		std::unique_ptr<Matrix<T>> create(size_t rowCount, size_t columnCount) override {
			std::vector<std::vector<T>> data(rowCount);
			for (auto &data_row : data) {
				data_row.resize(columnCount);
			}
			std::unique_ptr<Matrix<T>> retval{ new ReferenceMatrix<T>{ data } };
			return retval;
		};

		std::unique_ptr<Matrix<T>> create(std::initializer_list<std::initializer_list<T>> lists) override {
			std::vector<std::vector<T>> data;
			for (auto list : lists) {
				data.push_back(std::vector<T>{list});
			}
			std::unique_ptr<Matrix<T>> retval{ new ReferenceMatrix<T>{ data } };
			return retval;
		};

		std::unique_ptr<Matrix<T>> create(const std::vector<std::vector<T>> &v) override {
			std::unique_ptr<Matrix<T>> retval{ new ReferenceMatrix<T>{ v } };
			return retval;
		};

		std::unique_ptr<Matrix<T>> createRowMatrix(const std::vector<T> &t) override {
			std::vector<std::vector<T>> data{ t };
			std::unique_ptr<Matrix<T>> retval{ new ReferenceMatrix<T>{ data } };
			return retval;
		}

		std::string getInfo() override {
			return "Reference Matrix";
		}
	};
}
