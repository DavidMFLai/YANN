#pragma once
#include "Matrix.h"

namespace CPPANN {
	template<typename T>
	class ANN
	{
	public:
		ANN() = default;
		void addLayer(uint64_t size) {
			layers.push_back(Layer<T>{std::vector<T>(size)});
		}

		void addWeights(WeightMatrix<T> weightMatrix) {
			weights.push_back(std::move(weightMatrix));
		}

		const std::vector<T> &forwardPropagate(const std::vector<T> &input) {
			layers[0] = input;

			for (int i = 0; i < weights.size(); i++) {
				layers[i + 1] = weights[i] * layers[i];
			}
			for (int i = 0; i < layers.size(); i++) {
				layers[i].applySigmoid();
			}
			return layers.back().getValues();
		}

	private:
		std::vector<Layer<T>> layers;
		std::vector<WeightMatrix<T>> weights;
	};

}