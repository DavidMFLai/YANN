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
			if (signal_nodes.size() > 0) {
				network_nodes.push_back(Matrix<T>{1, size});
			}
			if (signal_nodes.size() > 1) {
				dSOutput_dSn.push_back(Matrix<T>{ size, signal_nodes.back().getDimensions()[1]});
			}
			signal_nodes.push_back(Matrix<T>{1, size});
		}

		void add_weights(Matrix<T> matrix) {
			weights.push_back(std::move(matrix));
			dSn_dWn_1.push_back(Matrix<T>{matrix.getDimensions()[1], matrix.getDimensions()[0]});
		}

		const std::vector<T> &forward_propagate(std::vector<T> &&input) {
			signal_nodes[0] = std::move(input);

			for (int i = 0; i < weights.size(); i++) {
				network_nodes[i] = signal_nodes[i] * weights[i];
				signal_nodes[i + 1] = sigmoid(network_nodes[i]);
			}

			return signal_nodes.back().getElems();
		}

		void back_propagate() {
			//compute dSOutput_dSn
			for (auto i = dSOutput_dSn.size() - 1; i != -1; --i) {
				auto dSn_dSn_1 = compute_dSn_dSn_1(network_nodes[i + 1], weights[i + 1]);
				if (i == dSOutput_dSn.size() - 1) {
					dSOutput_dSn[i] = std::move(dSn_dSn_1);
				}
				else {
					dSOutput_dSn[i] = dSOutput_dSn[i + 1] * dSn_dSn_1;
				}
			}


		}

	private:
		//returns retval(i,j) = d(signal_layer(0,i))/d(signal_layer_prev(0,j)).
		static Matrix<T> compute_dSn_dSn_1(const Matrix<T> &network_layer, const Matrix<T> &weights) {
			//output matrix has dimensions equal to weights transpose
			Matrix<T> retval{ weights.getDimensions()[1], weights.getDimensions()[0] };

			for (auto i = 0; i < retval.getDimensions()[0]; ++i) {
				//assuming sigmoid function
				auto dSn_dNn_1 = network_layer(0, i)*(1 - network_layer(0, i));
				for (auto j = 0; j < retval.getDimensions()[1]; ++j) {
					//Network to Signal layer
					auto dNn_1_dSn_1 = weights(j, i);

					retval(i, j) = dSn_dNn_1 * dNn_1_dSn_1;
				}
			}
			return retval;
		}

		//returns retval(i,j) := d(signal_layer(0,i))/d(weights_prev(j,i))
		static Matrix<T> compute_dSn_dWn_1(const Matrix<T> &network_layer, const Matrix<T> &signal_layer_prev) {
			Matrix<T> retval{ network_layer.getDimensions()[1], signal_layer_prev.getDimensions()[1] };

			for (auto i = 0; i < retval.getDimensions()[0]; ++i) {
				//assuming sigmoid function
				auto dSn_dNn_1 = network_layer(0, i)*(1 - network_layer(0, i));
				for (auto j = 0; j < retval.getDimensions()[1]; ++j) {
					//Network to Weight layer
					auto dNn_1_dWn_1 = signal_layer_prev(0, j);
					retval(i, j) = dSn_dNn_1 * dNn_1_dWn_1;
				}
			}
			return retval;
		}

	private:
		//dSOutput_dSn[n](i,j) := d(signal_nodes[last](0,i))/d(signal_nodes[n](0,j))
		std::vector<Matrix<T>> dSOutput_dSn;

		//dSn_dWn_1[n](i,j) := d(signal_nodes[n](0,i))/d(weights[n-1](j,i))
		std::vector<Matrix<T>> dSn_dWn_1;

		//Every signal_nodes[i], i!=0, represents a layer of neuron output values. signal_nodes[0] := input. All signal_nodes are row vectors
		std::vector<Matrix<T>> signal_nodes;

		//weights[m](i,j) := Weight from signal_nodes[m](0,i) to network_nodes[m](0,j). 
		std::vector<Matrix<T>> weights;

		//network_nodes[i] := signal_nodes[i] * weights[i]. All network_nodes are row vectors
		std::vector<Matrix<T>> network_nodes;
	};
}