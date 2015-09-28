#pragma once
#include <vector>
#include <array>
#include <cassert>

namespace CPPANN {
	template<typename T>
	class Layer
	{
	public:
		Layer(const std::vector<T> &initialValues) :
			networkValues{ std::move(initialValues) }
		{};

		const std::vector<T> &getValues() const
		{
			return networkValues;
		};

		void applySigmoid() {
			for (auto &networkValue : networkValues) {
				networkValue = sigmoid(networkValue);
			}
		};

	private:
		static T sigmoid(T x) {
			return 1 / (1 + exp(-x));
		};

		std::vector<T> networkValues;
	};


	template<typename T>
	class WeightMatrix
	{
	public:
		WeightMatrix(std::initializer_list<std::initializer_list<T>> inputRows) {
			for (auto &inputRow : inputRows) {
				auto weightsBegingIt = weights.end();
				weights.insert(weightsBegingIt, inputRow.begin(), inputRow.end());
			}
		}

		WeightMatrix(const std::vector<std::vector<T>> &inputRows) {
			for (auto &inputRow : inputRows) {
				auto weightsBegingIt = weights.end();
				weights.insert(weightsBegingIt, inputRow.begin(), inputRow.end());
			}
		}

		Layer<T> operator*(const Layer<T> &layer) {
			std::vector<T> retval;

			//initial conditions
			retval.push_back(0);
			std::vector<T> input = layer.getValues();
			auto inputIt = input.begin();

			//loop through the weights and produce a product
			for (auto dataIt = weights.begin(); dataIt != weights.end(); ++dataIt) {
				if (inputIt == input.end()) {
					inputIt = input.begin();
					retval.push_back(0);
				}
				retval.back() += (*inputIt)*(*dataIt);
				++inputIt;
			}
			return Layer<T>{std::move(retval)};
		}

	private:
		std::vector<T> weights;
	};


	template<typename T>
	class Matrix {
		struct MatrixAccessProperties {
			void setDimensions(size_t rowCount, size_t columnCount) {
				dimensions = { rowCount, columnCount };
			}

			size_t operator()(size_t i, size_t j) const{
				return i*dimensions[1] + j;
			}

			std::array<size_t, 2> dimensions;
		};

	public:
		//defaulted constructors and destructors
		Matrix() = default;
		Matrix(Matrix &&) = default;
		Matrix &operator=(Matrix &&) = default;
		Matrix(const Matrix &) = default;
		Matrix &operator=(const Matrix &) = default;
		~Matrix() = default;

		//Constructor by row and column size
		Matrix(size_t rowCount, size_t columnCount)
			:elems(rowCount*columnCount)	{
			matrixAccessProperties.setDimensions(rowCount, columnCount);
		};

		//Constructor by initialization list
		Matrix(std::initializer_list<std::initializer_list<T>> lists) {
			matrixAccessProperties.setDimensions(lists.size(), lists.begin()->size());
			for (std::initializer_list<T> list : lists) {
				elems.insert(elems.end(), list);
			}
		}

		//Getting dimensions
		std::array<size_t, 2> getDimensions() const{
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
}