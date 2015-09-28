#pragma once
#include "Matrix.h"

namespace CPPANN {
	template<typename T>
	class Layer : public Matrix<T>
	{
	public:
		Layer(size_t rowCount, size_t columnCount) 
			: Matrix<T>(rowCount, columnCount)
		{};

		Layer &operator=(const Matrix<T> & mat) {
			assert(this->getDimensions() == mat.getDimensions());
			Matrix<T>::operator=(mat);
			return *this;
		}

		void apply_sigmoid() {
			auto &elems = getElems();
			for (auto &elem : elems) {
				elem = sigmoid(elem);
			}
		};

		std::vector<T> &get_values() {
			return getElems();
		}

	private:
		static T sigmoid(T x) {
			return 1 / (1 + exp(-x));
		};
	};

	template<typename T>
	class Weight_Matrix : public Matrix<T> {
	public:
		Weight_Matrix(std::initializer_list<std::initializer_list<T>> lists)
			: Matrix(lists)
		{};
	};
	
	template<typename T>
	class ANN
	{
	public:
		ANN() = default;
		void add_layer(uint64_t size) {
			layers.push_back(Layer<T>{size, 1});
		}

		void add_weights(Weight_Matrix<T> matrix) {
			weights.push_back(std::move(matrix));
		}

		const std::vector<T> &forward_propagate(const std::vector<T> &input) {
			layers[0] = input;

			for (int i = 0; i < weights.size(); i++) {
				layers[i + 1] = weights[i] * layers[i];
				layers[i + 1].apply_sigmoid();
			}

			return layers.back().get_values();
		}

	private:
		std::vector<Layer<T>> layers;
		std::vector<Weight_Matrix<T>> weights;
	};

}