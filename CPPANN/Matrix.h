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




	template<int N>
	struct MatrixSlice {

	};

	template<>
	struct MatrixSlice<1> {
		MatrixSlice() = delete;
		MatrixSlice(size_t length)
			:size{length},
			extents{ length }

		{};

		size_t operator()(size_t i) const {
			return i;
		}

		const size_t size;
		const std::array<size_t, 1> extents;
	};

	template<>
	struct MatrixSlice<2> {
		MatrixSlice() = delete;
		MatrixSlice(size_t rowCount, size_t columnCount) 
			:stride{ columnCount },
			size{ rowCount * columnCount },
			extents{ rowCount, columnCount }
		{}

		size_t operator()(size_t i, size_t j) const{
			return i*stride + j;
		}

		const size_t size;
		const size_t stride;
		const std::array<size_t, 2> extents;
	};

	template<typename T, int N>
	class Matrix {
	public:
		Matrix() = default;
		Matrix(Matrix &&) = default;
		Matrix &operator=(Matrix &&) = default;
		Matrix(const Matrix &) = default;
		Matrix &operator=(const Matrix &) = default;
		~Matrix() = default;

		template<typename... Exts>
		Matrix(Exts... exts)
			:desc{ exts },
			elems(desc.getSize())
		{};

		std::array<size_t, N> extent() {
			return desc.extent;
		}

		template<typename... Args)
		T& operator()(Args... args) {
			auto slice = desc(args);
			return elems(slice);
		}

	private:
		MatrixSlice<N> desc;
		std::vector<T> elems;
	};


	// y = m*n; m is a M*P matrix, n is a column vector of length p
	template<typename T>  
	Matrix<T, 1> operator*(const Matrix<T, 2> &m, const Matrix<T, 1> &n) {
		assert(m.extent(1) == n.extent(0));
		Matrix<T, 1> retval(m.extent(0));
		for
	}
}