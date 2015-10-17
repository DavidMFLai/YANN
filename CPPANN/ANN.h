#pragma once
#include "Matrix.h"
#include <cmath>

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
	static T sigmoid_prime(T sigmoid_value) {
		return sigmoid_value * (1 - sigmoid_value);
	};

	template<typename T>
	static Matrix<T> tanh(Matrix<T> x) {
		Matrix<T> retval{ x.getDimensions()[0], x.getDimensions()[1] };
		for (auto i = 0; i < retval.getDimensions()[0]; ++i)
			for (auto j = 0; j < retval.getDimensions()[1]; ++j)
				retval(i, j) = std::tanh(-x(i, j));
		return retval;
	};

	template<typename T>
	static T tanh_prime(T tanh_value) {
		return 1 - tanh_value * tanh_value;
	};

	enum class Neuron_Type {
		Sigmoid,
		Tanh
	};

	template<typename T>
	static Matrix<T> evalute_perceptron(Matrix<T> x, Neuron_Type neuron_type) {
		if (neuron_type == Neuron_Type::Sigmoid) {
			return CPPANN::sigmoid<T>(std::move(x));
		}
		else {
			return CPPANN::tanh<T>(std::move(x));
		}	
	}

	template<typename T>
	static T evalute_perceptron_prime(T x, Neuron_Type neuron_type) {
		if (neuron_type == Neuron_Type::Sigmoid) {
			return sigmoid_prime(x);
		}
		else {
			return tanh_prime(x);
		}
	}

	template<typename T>
	class ANN {
	private:
		Neuron_Type neuron_type = Neuron_Type::Sigmoid;
	public:
		ANN() = default;
		void add_layer(uint64_t size) {
			if (signal_nodes.size() > 0) {
				network_nodes.push_back(Matrix<T>{1, size});
				dSnp1_dBn.push_back(Matrix<T>{size, network_nodes.back().getDimensions()[1] });
			}
			if (signal_nodes.size() > 1) {
				dSOutput_dSnp1.push_back(Matrix<T>{});
			}
			signal_nodes.push_back(Matrix<T>{1, size});
		}

		void add_bias(std::vector<T> &&input) {
			assert(network_nodes.size() > 0);
			assert(network_nodes.back().getDimensions()[1] == input.size());
			network_bias.push_back(std::move(input));
		}

		void add_weights(Matrix<T> matrix) {
			weights.push_back(std::move(matrix));
			weight_updates.push_back(Matrix<T>{matrix.getDimensions()[0], matrix.getDimensions()[1]});
			dSnp1_dWn.push_back(Matrix<T>{matrix.getDimensions()[1], matrix.getDimensions()[0]});
		}

		const std::vector<T> &forward_propagate(std::vector<T> &&input) {
			signal_nodes[0] = std::move(input);

			for (int i = 0; i < weights.size(); i++) {
				network_nodes[i] = signal_nodes[i] * weights[i] + network_bias[i];
				signal_nodes[i + 1] = evalute_perceptron(network_nodes[i], this->neuron_type);
			}

			return signal_nodes.back().getElems();
		}

		void back_propagate(const Matrix<T> &expected) {
			//speed
			const double speed = 0.5;

			//compute error
			error_vector = signal_nodes.back() - expected;

			//compute dSOutput_dSnp1.
			auto i = dSOutput_dSnp1.size() - 1;
			auto dSnp2_dSnp1 = compute_dSnp1_dSn(signal_nodes[i + 2], weights[i + 1], neuron_type);
			dSOutput_dSnp1[i] = std::move(dSnp2_dSnp1);
			for (auto i = (int64_t)(dSOutput_dSnp1.size() - 2); i >= 0; --i) {
				auto dSnp1_dSn = compute_dSnp1_dSn(signal_nodes[i+1], weights[i], neuron_type);
				dSOutput_dSnp1[i] = dSOutput_dSnp1[i + 1] * dSnp1_dSn;
			}

			//compute dSnp1_dWn
			for (auto i = 0; i < dSnp1_dWn.size(); ++i) {
				auto dSnp1_dWn_single = compute_dSnp1_dWn(signal_nodes[i + 1], signal_nodes[i], neuron_type);
				dSnp1_dWn[i] = std::move(dSnp1_dWn_single);
			}

			//compute weight updates
			for (size_t indexOfNodeInSOutput = 0; indexOfNodeInSOutput < signal_nodes.back().getDimensions()[1]; indexOfNodeInSOutput++) {
				std::vector<Matrix<T>> outputnode_contributions = compute_dSOutput_dWn(dSOutput_dSnp1, indexOfNodeInSOutput, dSnp1_dWn, error_vector(0, indexOfNodeInSOutput));
				for (size_t idx = 0; idx < outputnode_contributions.size(); ++idx) {
					weights[idx] -= (outputnode_contributions[idx] * speed);
				}
			}

			//compute dSnp1_dBn		
			for (size_t i = 0; i < dSnp1_dBn.size(); ++i) {
				auto dSnp1_dBn_single = compute_dSnp1_dBn(signal_nodes[i+1], neuron_type);
				dSnp1_dBn[i] = std::move(dSnp1_dBn_single);
			}

			//compute bias updates
			for (size_t indexOfNodeInSOutput = 0; indexOfNodeInSOutput < signal_nodes.back().getDimensions()[1]; indexOfNodeInSOutput++) {
				std::vector<Matrix<T>> outputnode_contributions = compute_dOutput_dBn(dSOutput_dSnp1, indexOfNodeInSOutput, dSnp1_dBn, error_vector(0, indexOfNodeInSOutput));
				for (size_t idx = 0; idx < outputnode_contributions.size(); ++idx) {
					network_bias[idx] -= outputnode_contributions[idx] * speed;
				}
			}
		}

	private:
		//returns retval(i,j) = d(signal_layer_next(0,i))/d(signal_layer(0,j)).
		static Matrix<T> compute_dSnp1_dSn(const Matrix<T> &signal_layer_next, const Matrix<T> &weights, Neuron_Type neuron_type) {
			//output matrix has dimensions equal to weights transpose
			Matrix<T> retval{ weights.getDimensions()[1], weights.getDimensions()[0] };

			for (auto i = 0; i < retval.getDimensions()[0]; ++i) {
				auto dSnp1_dNn = evalute_perceptron_prime(signal_layer_next(0, i), neuron_type);
				for (auto j = 0; j < retval.getDimensions()[1]; ++j) {
					//Network to Signal layer
					auto dNn_dSn = weights(j, i);

					retval(i, j) = dSnp1_dNn * dNn_dSn;
				}
			}
			return retval;
		}

		//returns retval(i,j) := d(signal_layer_next(0,i))/d(weights(j,i))
		static Matrix<T> compute_dSnp1_dWn(const Matrix<T> &signal_layer_next, const Matrix<T> &signal_layer, Neuron_Type neuron_type) {
			Matrix<T> retval{ signal_layer_next.getDimensions()[1], signal_layer.getDimensions()[1] };

			for (auto i = 0; i < retval.getDimensions()[0]; ++i) {
				auto dSnp1_dNn = evalute_perceptron_prime(signal_layer_next(0, i), neuron_type);
				for (auto j = 0; j < retval.getDimensions()[1]; ++j) {
					//Network to Weight layer
					auto dNn_dWn = signal_layer(0, j);
					retval(i, j) = dSnp1_dNn * dNn_dWn;
				}
			}
			return retval;
		}

		//returns dSOutput[indexOfNodeInSOutput]/dWn(). Hence, e.g. retval[10](20,30) = dSOutput_indexOfNodeInSOutput/dW[10](30,20)
		static std::vector<Matrix<T>> compute_dSOutput_dWn(const std::vector<Matrix<T>> &dSOutput_dSnp1, size_t indexOfNodeInSOutput, const std::vector<Matrix<T>> &dSnp1_dWn, double error) {
			std::vector<Matrix<T>> retval;
			for (size_t idx = 0; idx < dSnp1_dWn.size()-1; ++idx) {
				Matrix<T> dSOutput_dWn_single{ dSnp1_dWn[idx].getDimensions()[1], dSnp1_dWn[idx].getDimensions()[0] };
				for (size_t i = 0; i < dSOutput_dWn_single.getDimensions()[0]; ++i)
					for (size_t j = 0; j < dSOutput_dWn_single.getDimensions()[1]; ++j)
						dSOutput_dWn_single(i, j) = dSOutput_dSnp1[idx](indexOfNodeInSOutput, j)*dSnp1_dWn[idx](j, i) * error;
				retval.push_back(std::move(dSOutput_dWn_single));
			}
			retval.push_back(dSnp1_dWn[dSnp1_dWn.size() - 1].createRowMatrix(indexOfNodeInSOutput).transpose() * error);
			return retval;
		}

		//returns retval(0,i) := d(signal_layer_next(0,i))/d(network_bias(0,j))
		static Matrix<T> compute_dSnp1_dBn(const Matrix<T> &signal_layer_next, Neuron_Type neuron_type) {
			Matrix<T> retval{ signal_layer_next.getDimensions()[0], signal_layer_next.getDimensions()[1] };

			for (auto i = 0; i < retval.getDimensions()[1]; ++i) {
				auto dSnp1_dNn = evalute_perceptron_prime(signal_layer_next(0, i), neuron_type);
				retval(0, i) = dSnp1_dNn; //because dNn/dBn = 1
			}

			return retval;
		}

		//returns dOutput[indexOfNodeInSOutput]/dBn(), e.g. retval[10](0,30) = dSOutput_indexOfNodeInSOutput/dBias[10](0, 30). Notice that output is a std::vector of row vectors prese.
		static std::vector<Matrix<T>> compute_dOutput_dBn(const std::vector<Matrix<T>> &dSOutput_dSnp1, size_t indexOfNodeInSOutput, const std::vector<Matrix<T>> &dSnp1_dBn, double error) {
			std::vector<Matrix<T>> retval;
			for (size_t layer_index = 0; layer_index < dSnp1_dBn.size() - 1; ++layer_index) {
				Matrix<T> retval_single{ 1, dSnp1_dBn[layer_index].getDimensions()[1] };
				for (size_t indexofNodeInBias = 0; indexofNodeInBias < dSnp1_dBn[layer_index].getDimensions()[1]; ++indexofNodeInBias) {
					retval_single(0, indexofNodeInBias) = dSOutput_dSnp1[layer_index](indexOfNodeInSOutput, indexofNodeInBias) * dSnp1_dBn[layer_index](0, indexofNodeInBias) * error;
				}
				retval.push_back(retval_single);
			}
			retval.push_back(dSnp1_dBn[dSnp1_dBn.size() - 1] * error);

			return retval;
		}

	private:
		/*
		Back propagation variables:
		*/
			Matrix<T> error_vector;

			//dSOutput_dSnp1[n](i,j) := d(signal_nodes[last](0,i))/d(signal_nodes[n](0,j))
			std::vector<Matrix<T>> dSOutput_dSnp1;

			//dSnp1_dWn[n](i,j) := d(signal_nodes[n+1](0,i))/d(weights[n](j,i))
			std::vector<Matrix<T>> dSnp1_dWn;

			//dSnp1_dBn[n](i,j) := d(signal_nodes[n+1](0,i))/d(network_bias[n](0,j))
			std::vector<Matrix<T>> dSnp1_dBn;

			//Resultant updates
			std::vector<Matrix<T>> weight_updates;
			std::vector<Matrix<T>> network_bias_updates;
		
		/*
		Forward propagation variables:
		*/
			//Every signal_nodes[i], i!=0, represents a layer of neuron output values. signal_nodes[0] := input. All signal_nodes are row vectors
			std::vector<Matrix<T>> signal_nodes;

			//weights[m](i,j) := Weight from signal_nodes[m](0,i) to network_nodes[m](0,j). 
			std::vector<Matrix<T>> weights;

			//network_nodes[i] := signal_nodes[i] * weights[i]. All network_nodes are row vectors
			std::vector<Matrix<T>> network_nodes;

			//network_bias[i] := bias values on the neurons
			std::vector<Matrix<T>> network_bias;
	
	public:

		const std::vector<Matrix<T>> &getWeights() const{
			return weights;
		};

		const std::vector<Matrix<T>> &getBiases() const{
			return network_bias;
		};

	};
}