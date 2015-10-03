#pragma once
#include "Matrix.h"

namespace CPPANN {
	template<typename T>
	static Matrix<T> sigmoid(Matrix<T> x) {
		Matrix<T> retval{ x.getDimensions()[0], x.getDimensions()[1] };
		for (auto i = 0; i < retval.getDimensions()[0]; ++i)
			for (auto j = 0; j < retval.getDimensions()[1]; ++j)
				retval(i, j) = 1 / (1 + exp(-x(i,j)));
		return retval;
	};

	template<typename T>
	class ANN {
	public:
		ANN() = default;
		void add_layer(uint64_t size) {
			signal_nodes.push_back(Matrix<T>{1, size});
			if (signal_nodes.size() > 1) {
				network_nodes.push_back(Matrix<T>{1, size});
			}
		}

		void add_weights(Matrix<T> matrix) {
			weights.push_back(std::move(matrix));
		}

		const std::vector<T> &forward_propagate(std::vector<T> &&input) {
			signal_nodes[0] = std::move(input);

			for (int i = 0; i < weights.size(); i++) {
				network_nodes[i] = signal_nodes[i] * weights[i];
				signal_nodes[i + 1] = sigmoid(network_nodes[i]);
			}

			return signal_nodes.back().getElems();
		}

	public:
		Matrix<T> compute_dSn_dSn_1(const Matrix<T> &nodes, const Matrix<T> &weights){
			//find dimensions
			auto dimensions_of_weight_matrix = weights.getDimensions();

			//output matrix has dimensions equal to weights transpose
			Matrix<T> retval{ dimensions_of_weight_matrix[1], dimensions_of_weight_matrix[0] };

			for (auto i = 0; i < retval.getDimensions()[0]; ++i)
				for (auto j = 0; j < retval.getDimensions()[1]; ++j) {
					//assuming sigmoid function
					dSn_dNn_1 = nodes(0, i)*(1 - nodes(0, i));

					//transfer from Network to Signal layer
					dNn_1_dSn_1 = weights[j][i];

					retval(i, j) = dSn_dNn_1 * dNn_1_dSn_1;
				}
			return retval;
		}

	private:
		std::vector<Matrix<T>> dSOutput_dSn;

		std::vector<Matrix<T>> signal_nodes;
		std::vector<Matrix<T>> weights;
		std::vector<Matrix<T>> network_nodes;
	};

}