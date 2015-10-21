#pragma once
#include "Matrix.h"
#include <cmath>

namespace CPPANN {
	template<typename T>
	static void sigmoid(Matrix<T> &output, const Matrix<T> &x) {
		assert(output.getDimensions() == x.getDimensions());
		for (auto i = 0; i < output.getDimensions()[0]; ++i)
			for (auto j = 0; j < output.getDimensions()[1]; ++j)
				output(i, j) = 1 / (1 + std::exp(-x(i,j)));
	};

	template<typename T>
	static T sigmoid_prime(T sigmoid_value) {
		return sigmoid_value * (1 - sigmoid_value);
	};

	template<typename T>
	static void tanh(Matrix<T> &output, const Matrix<T> &x) {
		assert(output.getDimensions() == x.getDimensions());
		for (auto i = 0; i < output.getDimensions()[0]; ++i)
			for (auto j = 0; j < output.getDimensions()[1]; ++j)
				output(i, j) = std::tanh(-x(i, j));
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
	static void evalute_perceptron(Matrix<T> &output, const Matrix<T> &input, Neuron_Type neuron_type) {
		if (neuron_type == Neuron_Type::Sigmoid) {
			CPPANN::sigmoid<T>(output, input);
		}
		else {
			CPPANN::tanh<T>(output, input);
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
	class ANN;

	template<typename T>
	class ANNBuilder {
	public:
		ANNBuilder() = default;

		ANNBuilder &set_layer(size_t layer_index, uint64_t size) {
			if (neuron_counts.size() <= layer_index) {
				neuron_counts.resize(layer_index + 1);
			}
			neuron_counts.at(layer_index) = size;
			return *this;
		}
		ANNBuilder &set_bias(size_t layer_index, const std::vector<T> &input) {
			if (biases_of_each_layer.size() <= layer_index) {
				biases_of_each_layer.resize(layer_index + 1);
			}
			biases_of_each_layer.at(layer_index) = input;
			return *this;
		}
		ANNBuilder &set_weights(size_t starting_layer_index, const Matrix<T> &matrix) {
			if (weight_matrices.size() <= starting_layer_index) {
				weight_matrices.resize(starting_layer_index + 1);
			}
			weight_matrices.at(starting_layer_index) = matrix;
			return *this;
		}
		ANN<T> build() {
			ANN<T> retval{ neuron_counts, biases_of_each_layer, weight_matrices };
			return retval;
		}
	private:
		std::vector<size_t> neuron_counts;
		std::vector<std::vector<T>> biases_of_each_layer;
		std::vector<Matrix<T>> weight_matrices;

	};

	template<typename T>
	class ANN {
	private:
		Neuron_Type neuron_type = Neuron_Type::Sigmoid;
		friend ANN<T> ANNBuilder<T>::build();
		ANN(const std::vector<size_t> &signal_counts, const std::vector<std::vector<T>> &biases_of_each_layer, const std::vector<Matrix<T>> &weight_matrices) {

			//Get number of layers
			size_t layers_count = signal_counts.size();
			
			//Create matrices required by forward propagation 
			signal_nodes.resize(layers_count);
			network_nodes.resize(layers_count - 1);
			network_bias.resize(layers_count - 1);
			weights.resize(layers_count - 1);
			for (size_t idx = 0; idx < layers_count; idx++) {
				if (idx == 0) {
					signal_nodes.at(idx) = Matrix<T>{1, signal_counts[idx]};
				}
				else {
					signal_nodes.at(idx) = Matrix<T>{1, signal_counts[idx]};
					network_nodes.at(idx - 1) = Matrix<T>{1, signal_counts[idx]};
					network_bias.at(idx - 1) = biases_of_each_layer.at(idx - 1);
					weights.at(idx - 1) = weight_matrices.at(idx - 1);
				}
			}

			//Create additional matrices required by backward propagation
			dSnp1_dBn.resize(layers_count - 1);
			dSnp1_dWn.resize(layers_count - 1);
			dSOutput_dSnp1.resize(layers_count - 2);
			dSnp2_dSnp1.resize(layers_count - 2);
			for (size_t idx = 0; idx < layers_count - 1; idx++) {
				if (idx == 0) {
					dSnp1_dBn.at(idx) = Matrix<T>{ 1, network_nodes.at(idx).getDimensions()[1] };
					dSnp1_dWn.at(idx) = Matrix<T>{ weights.at(idx + 1).getDimensions()[1],  weights.at(idx).getDimensions()[0] };
				}
				else {
					dSnp1_dBn.at(idx) = Matrix<T>{ 1, network_nodes.at(idx).getDimensions()[1] };
					dSnp1_dWn.at(idx) = Matrix<T>{ weights.at(idx).getDimensions()[1],  weights.at(idx).getDimensions()[0] };
					dSOutput_dSnp1.at(idx - 1) = Matrix<T>{signal_nodes.back().getDimensions()[1], signal_nodes.at(idx - 1).getDimensions()[1]};
					dSnp2_dSnp1.at(idx - 1) = Matrix<T>{ signal_nodes.at(idx + 1).getDimensions()[1], signal_nodes.at(idx).getDimensions()[1] };
				}
			}

		}
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
				Matrix<T>::multiply(network_nodes[i], signal_nodes[i], weights[i]);
				Matrix<T>::add(network_nodes[i], network_nodes[i], network_bias[i]);
				evalute_perceptron(signal_nodes[i + 1], network_nodes[i], this->neuron_type);
			}
			return signal_nodes.back().getElems();
		}

		void back_propagate(const Matrix<T> &expected) {
			//speed
			const double speed = 0.5;

			//compute error
			error_vector = signal_nodes.back() - expected;

			//compute dSnp2_dSnp1
			for (size_t idx = 0; idx < dSnp2_dSnp1.size(); ++idx) {
				compute_dSnp1_dSn(dSnp2_dSnp1[idx], signal_nodes[idx + 2], weights[idx + 1], neuron_type);
			}

			//compute dSOutput_dSnp1
			dSOutput_dSnp1.back() = dSnp2_dSnp1.back();
			if (dSOutput_dSnp1.size() > 1) {
				for (size_t idx = dSOutput_dSnp1.size() - 2; idx >= 0; --idx) {
					Matrix<T>::multiply(dSOutput_dSnp1[idx], dSOutput_dSnp1[idx + 1], dSnp2_dSnp1[idx]);
				}
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
		static void compute_dSnp1_dSn(Matrix<T> &output, const Matrix<T> &signal_layer_next, const Matrix<T> &weights, Neuron_Type neuron_type) {
			for (auto i = 0; i < output.getDimensions()[0]; ++i) {
				auto dSnp1_dNn = evalute_perceptron_prime(signal_layer_next(0, i), neuron_type);
				for (auto j = 0; j < output.getDimensions()[1]; ++j) {
					//Network to Signal layer
					auto dNn_dSn = weights(j, i);
					output(i, j) = dSnp1_dNn * dNn_dSn;
				}
			}
		}

		//returns retval(i,j) = d(signal_layer_next(0,i))/d(signal_layer(0,j)).
		static Matrix<T> deprecated_compute_dSnp1_dSn(const Matrix<T> &signal_layer_next, const Matrix<T> &weights, Neuron_Type neuron_type) {
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

			//dSnp2_dSnp1[n](i,j) := d(signal_nodes[n+2](0,i))/d(signal_nodes[n+1](0,j))
			std::vector<Matrix<T>> dSnp2_dSnp1;

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