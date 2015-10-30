#pragma once
#include "Matrix.h"
#include <cmath>
#include <cassert>
#include <random>
#include <string>

using std::string;
namespace CPPANN {
	template<typename T>
	static void Sigmoid(Matrix<T> &output, const Matrix<T> &x) {
		assert(output.getDimensions() == x.getDimensions());
		for (size_t idx = 0; idx < x.getElems().size(); ++idx) {
			output.getElems()[idx] = 1 / (1 + std::exp(-x.getElems()[idx]));
		}
	};

	template<typename T>
	static T Sigmoid_prime(T sigmoid_value) {
		return sigmoid_value * (1 - sigmoid_value);
	};

	template<typename T>
	static void Tanh(Matrix<T> &output, const Matrix<T> &x) {
		assert(output.getDimensions() == x.getDimensions());
		for (size_t idx = 0; idx < x.getElems().size(); ++idx) {
			output.getElems()[idx] = std::tanh(x.getElems()[idx]);
		}
	};

	template<typename T>
	static T Tanh_prime(T tanh_value) {
		return 1 - tanh_value * tanh_value;
	};

	enum class Neuron_Type {
		Sigmoid,
		Tanh
	};

	template<typename T>
	static void evalute_perceptron(Matrix<T> &output, const Matrix<T> &input, Neuron_Type neuron_type) {
		if (neuron_type == Neuron_Type::Sigmoid) {
			CPPANN::Sigmoid<T>(output, input);
		}
		else {
			CPPANN::Tanh<T>(output, input);
		}	
	}

	template<typename T>
	static T evalute_perceptron_prime(T x, Neuron_Type neuron_type) {
		if (neuron_type == Neuron_Type::Sigmoid) {
			return Sigmoid_prime(x);
		}
		else {
			return Tanh_prime(x);
		}
	}

	template<typename T>
	class ANN;

	template<typename T >
	class Random_number_generator {
	public:
		Random_number_generator()
			: gen{ rd() }, dist{ 0, 0.1 }
		{}

		T generate() const{
			return dist(gen);
		};

	private:
		std::random_device rd;
		mutable std::mt19937 gen;
		mutable std::normal_distribution<T> dist;
	};

	template<typename T>
	class ANNBuilder {
	public:
		ANNBuilder()
			: output_neuron_count{ 0 } 
		{}

		ANNBuilder &set_input_layer(uint64_t size) {
			if (neuron_counts.size() == 0) {
				neuron_counts.resize(1);
			}
			neuron_counts.at(0) = size;
			return *this;
		}

		//Hidden layer index is zero based
		ANNBuilder &set_hidden_layer(size_t hidden_layer_index, Neuron_Type type, T speed, uint64_t size) {
			//setup neuron_counts
			//0th hidden layer is the 1st layer in the ANN. (0th layer in the ANN is the input layer)
			size_t layer_index = hidden_layer_index + 1;
			if (neuron_counts.size() <= layer_index) {
				neuron_counts.resize(layer_index + 1);
			}
			neuron_counts.at(layer_index) = size;

			//setup neuron_types
			if (neuron_types.size() <= hidden_layer_index) {
				neuron_types.resize(hidden_layer_index + 1);
			}
			neuron_types.at(hidden_layer_index) = type;

			//setup speeds
			if (speeds.size() <= hidden_layer_index) {
				speeds.resize(hidden_layer_index + 1);
			}
			speeds.at(hidden_layer_index) = speed;

			return *this;
		}

		ANNBuilder &set_output_layer(Neuron_Type type, T speed, uint64_t size) {
			output_neuron_type = type;
			output_speed = speed;
			output_neuron_count = size;
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
			if (output_neuron_count != 0) {
				neuron_counts.push_back(output_neuron_count);
			}
			else {
				throw runtime_error("output neuron has not been set");
			}
			neuron_types.push_back(output_neuron_type);
			speeds.push_back(output_speed);

			//fix biases and weights. this is needed because the user might not have correctly input the biases and weights
			Make_random_biases_if_needed(biases_of_each_layer, neuron_counts, random_number_generator);
			Make_random_weights_if_needed(weight_matrices, neuron_counts, random_number_generator);

			ANN<T> retval{ neuron_counts, biases_of_each_layer, weight_matrices, neuron_types, speeds};
			return retval;
		}

		const std::vector<size_t> &get_neuron_counts() const {
			return neuron_counts;
		}
		 
		const std::vector<std::vector<T>> &get_biases_of_each_layer() const {
			return biases_of_each_layer;
		}

		const std::vector<Matrix<T>> &get_weight_matrices() const {
			return weight_matrices;
		}

		const std::vector<Neuron_Type> &get_neuron_types() const {
			return neuron_types;
		}

		const std::vector<T> &get_speeds() const {
			return speeds;
		}

	private:
		static bool Neuron_count_biases_count_mismatch(const std::vector<size_t> &neuron_counts, const std::vector<std::vector<T>> &biases_of_each_layer, size_t layer_idx) {
			return neuron_counts.at(layer_idx + 1) != biases_of_each_layer.at(layer_idx).size();
		}

		static bool Neuron_count_weights_count_mismatch(const std::vector<size_t> &neuron_counts, const std::vector<Matrix<T>> &weight_matrices, size_t layer_idx) {
			bool retval = false;
			if (weight_matrices.at(layer_idx).getRowCount() != neuron_counts.at(layer_idx)) {
				retval = true;
			}
			else if (weight_matrices.at(layer_idx).getColumnCount() != neuron_counts.at(layer_idx+1)) {
				retval = true;
			}
			return retval;
		}

		static void Make_random_biases_if_needed(std::vector<std::vector<T>> &biases_of_each_layer, const std::vector<size_t> &neuron_counts, const Random_number_generator<T> &random_number_generator) {
			if (biases_of_each_layer.size() != neuron_counts.size() - 1) {
				biases_of_each_layer.resize(neuron_counts.size() - 1);
			}
			for (size_t idx = 0; idx < biases_of_each_layer.size(); idx++) {
				if (Neuron_count_biases_count_mismatch(neuron_counts, biases_of_each_layer, idx)) {
					//create a new bias vector of length == neuron_counts.at(idx + 1)
					std::vector<T> random_biases;
					random_biases.resize(neuron_counts.at(idx + 1 ));
					for (size_t j = 0; j < random_biases.size(); ++j) {
						random_biases.at(j) = random_number_generator.generate();
					}

					//put the newly created vector into 
					biases_of_each_layer.at(idx) = random_biases;
				}
			}
		}

		static void Make_random_weights_if_needed(std::vector<Matrix<T>> &weight_matrices, const std::vector<size_t> &neuron_counts, const Random_number_generator<T> &random_number_generator) {
			if (weight_matrices.size() != neuron_counts.size() - 1) {
				weight_matrices.resize(neuron_counts.size() - 1);
			}
			for (size_t idx = 0; idx < weight_matrices.size(); idx++) {
				if (Neuron_count_weights_count_mismatch(neuron_counts, weight_matrices, idx)) {
					//create a new Matrix
					Matrix<T> random_matrix { neuron_counts.at(idx), neuron_counts.at(idx+1) };
					for (size_t i = 0; i < neuron_counts.at(idx); i++) {
						for (size_t j = 0; j < neuron_counts.at(idx + 1); j++) {
							random_matrix(i, j) = random_number_generator.generate();
						}
					}
					weight_matrices.at(idx) = random_matrix;
				}
			}
		}

		size_t output_neuron_count;
		Neuron_Type output_neuron_type;
		T output_speed;
		
		/*
		Values to be given to the ANN
		*/
		Random_number_generator<T> random_number_generator;
		std::vector<size_t> neuron_counts;
		std::vector<Neuron_Type> neuron_types;
		std::vector<T> speeds;
		std::vector<std::vector<T>> biases_of_each_layer;
		std::vector<Matrix<T>> weight_matrices;
	};

	template<typename T>
	class ANN {
	private:
		friend ANN<T> ANNBuilder<T>::build();
		ANN(const std::vector<size_t> &signal_counts, 
			const std::vector<std::vector<T>> &biases_of_each_layer, 
			const std::vector<Matrix<T>> &weight_matrices, 
			const std::vector<Neuron_Type> &neuron_types,
			const std::vector<T> &speeds
			) {

			//Get number of layers
			size_t layers_count = signal_counts.size();
			
			//set neuron types
			this->neuron_types = neuron_types;

			//set speeds
			this->speeds = speeds;

			//Create matrices required by forward propagation 
			signal_nodes.resize(layers_count);
			network_nodes.resize(layers_count - 1);
			biases.resize(layers_count - 1);
			weights.resize(layers_count - 1);
			
			for (size_t idx = 0; idx < layers_count; idx++) {
				if (idx == 0) {
					signal_nodes.at(idx) = Matrix<T>{1, signal_counts[idx]};
				}
				else {
					signal_nodes.at(idx) = Matrix<T>{1, signal_counts[idx]};
					network_nodes.at(idx - 1) = Matrix<T>{1, signal_counts[idx]};
					biases.at(idx - 1) = biases_of_each_layer.at(idx - 1);
					weights.at(idx - 1) = weight_matrices.at(idx - 1);
				}
			}

			//Create additional matrices required by backward propagation
			dSnp1_dBn.resize(layers_count - 1);
			dSnp1_dWn.resize(layers_count - 1);
			weight_updates.resize(layers_count - 1);
			bias_updates.resize(layers_count - 1);
			dSOutput_dSnp1.resize(layers_count - 2);
			dError_dSnp1.resize(layers_count - 2);
			dSnp2_dSnp1.resize(layers_count - 2);
			dTotalError_dSnp1.resize(layers_count - 2);
			error_vector = Matrix<T>{ signal_nodes.back().getRowCount(), signal_nodes.back().getColumnCount() };

			for (size_t idx = 0; idx < layers_count - 1; idx++) {
				if (idx == 0) {
					dSnp1_dBn.at(idx) = Matrix<T>{ 1, network_nodes.at(idx).getColumnCount() };
					dSnp1_dWn.at(idx) = Matrix<T>{ weights.at(idx).getRowCount(),  weights.at(idx).getColumnCount() };
					weight_updates.at(idx) = Matrix<T>{ weights.at(idx).getRowCount(), weights.at(idx).getColumnCount() };
					bias_updates.at(idx) = Matrix<T>{ biases.at(idx).getRowCount(), biases.at(idx).getColumnCount() };
				}
				else {
					dSnp1_dBn.at(idx) = Matrix<T>{ 1, network_nodes.at(idx).getColumnCount() };
					dSnp1_dWn.at(idx) = Matrix<T>{ weights.at(idx).getRowCount(),  weights.at(idx).getColumnCount() };
					weight_updates.at(idx) = Matrix<T>{ weights.at(idx).getRowCount(), weights.at(idx).getColumnCount() };
					bias_updates.at(idx) = Matrix<T>{ biases.at(idx).getRowCount(), biases.at(idx).getColumnCount() };
					dSOutput_dSnp1.at(idx - 1) = Matrix<T>{signal_nodes.back().getColumnCount(), signal_nodes.at(idx).getColumnCount()};
					dError_dSnp1.at(idx - 1) = Matrix<T>{ signal_nodes.back().getColumnCount(), signal_nodes.at(idx).getColumnCount() };
					dTotalError_dSnp1.at(idx - 1) = Matrix<T>{ 1, signal_nodes.at(idx).getColumnCount() };
				}
			}

			for (size_t idx = 1; idx < layers_count - 2; idx++) {
				dSnp2_dSnp1.at(idx - 1) = Matrix<T>{ signal_nodes.at(idx + 1).getColumnCount(), signal_nodes.at(idx).getColumnCount() };
			}
		}
	public:
		const std::vector<T> &forward_propagate(const std::vector<T> &input) {
			Matrix<T>::copy_from_vector(signal_nodes[0], input);
			for (int i = 0; i < weights.size(); i++) {
				Matrix<T>::Multiply(network_nodes[i], signal_nodes[i], weights[i]);
				Matrix<T>::Add(network_nodes[i], network_nodes[i], biases[i]);
				evalute_perceptron(signal_nodes[i + 1], network_nodes[i], this->neuron_types.at(i));
			}
			return signal_nodes.back().getElems();
		}

		void back_propagate(const Matrix<T> &expected) {
			//compute error
			Matrix<T>::Minus(error_vector, signal_nodes.back(), expected);

			//compute dSnp2_dSnp1
			for (size_t idx = 0; idx < dSnp2_dSnp1.size() - 1; ++idx) {
				Compute_dSnp1_dSn(dSnp2_dSnp1[idx], signal_nodes[idx + 2], weights[idx + 1], neuron_types.at(idx+1));
			}

			//compute dSOutput_dSnp1
			Compute_dSnp1_dSn(dSOutput_dSnp1.back(), signal_nodes.back(), weights.back(), neuron_types.back());
			if (dSOutput_dSnp1.size() >= 2) {
				for (size_t idx = dSOutput_dSnp1.size() - 2; idx != std::numeric_limits<size_t>::max(); --idx) {
					Matrix<T>::Multiply(dSOutput_dSnp1[idx], dSOutput_dSnp1[idx + 1], dSnp2_dSnp1[idx]);
				}
			}

			//compute dSnp1_dWn
			for (auto idx = 0; idx < dSnp1_dWn.size(); ++idx) {
				Compute_dSnp1_dWn(dSnp1_dWn[idx], signal_nodes[idx + 1], signal_nodes[idx], neuron_types.at(idx));
			}

			//compute dSError_dSnp1
			for (size_t idx = 0; idx < dError_dSnp1.size(); ++idx) {
				for (size_t i = 0; i < dError_dSnp1[idx].getRowCount(); ++i) {
					for (size_t j = 0; j < dError_dSnp1[idx].getColumnCount(); ++j)
						dError_dSnp1[idx](i, j) = dSOutput_dSnp1[idx](i, j)*error_vector(0, i);
				}
			}

			// compute weight updates and apply them
			Compute_dTotalError_dSnp1(dTotalError_dSnp1, dError_dSnp1);
			for (size_t idx = 0; idx < weight_updates.size()-1; ++idx) {
				Compute_weight_update(weight_updates[idx], dTotalError_dSnp1[idx], dSnp1_dWn[idx], speeds[idx]);
				weights[idx] -= weight_updates[idx];
			}
			Compute_final_weight_update(weight_updates.back(), error_vector, dSnp1_dWn.back(), speeds.back());
			weights.back() -= weight_updates.back();

			//compute dSnp1_dBn		
			for (size_t idx = 0; idx < dSnp1_dBn.size(); ++idx) {
				Compute_dSnp1_dBn(dSnp1_dBn[idx], signal_nodes[idx+1], neuron_types.at(idx));
			}

			//compute bias updates and apply them
			for (size_t idx = 0; idx < bias_updates.size() - 1; ++idx) {
				Compute_bias_update(bias_updates[idx], dTotalError_dSnp1[idx], dSnp1_dBn[idx], speeds[idx]);
				biases[idx] -= bias_updates[idx];
			}
			Compute_final_bias_update(bias_updates.back(), error_vector, dSnp1_dBn.back(), speeds.back());
			biases.back() -= bias_updates.back();
		}

	private:
		//returns output(i,j) = d(signal_layer_next(0,i))/d(signal_layer(0,j)).
		static void Compute_dSnp1_dSn(Matrix<T> &output, const Matrix<T> &signal_layer_next, const Matrix<T> &weights, Neuron_Type neuron_type) {
			assert(output.getRowCount() == signal_layer_next.getColumnCount());
			assert(output.getRowCount() == weights.getColumnCount());
			assert(output.getColumnCount() == weights.getRowCount());
			for (auto i = 0; i < output.getRowCount(); ++i) {
				auto dSnp1_dNn = evalute_perceptron_prime(signal_layer_next(0, i), neuron_type);
				for (auto j = 0; j < output.getColumnCount(); ++j) {
					//Network to Signal layer
					auto dNn_dSn = weights(j, i);
					output(i, j) = dSnp1_dNn * dNn_dSn;
				}
			}
		}

		//returns dTotalError_dSnp1(0,j) := d(sum(error_vector))/d(Snp1(0,j)) 
		static void Compute_dTotalError_dSnp1(std::vector<Matrix<T>> &dTotalError_dSnp1, const std::vector<Matrix<T>> &dSOutput_dSnp1) {
			assert(dTotalError_dSnp1.size() == dSOutput_dSnp1.size());
			for (size_t idx = 0; idx < dTotalError_dSnp1.size(); ++idx) {
				Matrix<T>::Sum_of_rows(dTotalError_dSnp1[idx], dSOutput_dSnp1[idx]);
			}
		}

		//returns output(i,j) := d(signal_layer_next(0,j))/d(weights(i,j))
		static void Compute_dSnp1_dWn(Matrix<T> &output, const Matrix<T> &signal_layer_next, const Matrix<T> &signal_layer, Neuron_Type neuron_type) {
			assert(output.getRowCount() == signal_layer.getColumnCount());
			assert(output.getColumnCount() == signal_layer_next.getColumnCount());
			for (auto i = 0; i < output.getRowCount(); ++i) {
				//Network to Weight layer
				auto dNn_dWn = signal_layer(0, i);
				for (auto j = 0; j < output.getColumnCount(); ++j) {
					auto dSnp1_dNn = evalute_perceptron_prime(signal_layer_next(0, j), neuron_type);
					output(i, j) = dSnp1_dNn * dNn_dWn;
				}
			}
		}

		//returns the weight_update of the Nth layer
		static void Compute_weight_update(Matrix<T> &weight_update_n, const Matrix<T> &dTotalError_dSnp1, const Matrix<T> &dSnp1_dWn, T speed) {
			assert(weight_update_n.getDimensions() == dSnp1_dWn.getDimensions());
			assert(weight_update_n.getColumnCount() == dSnp1_dWn.getColumnCount());
			for (size_t i = 0; i < weight_update_n.getRowCount(); i++) 
				for (size_t j = 0; j < weight_update_n.getColumnCount(); j++) {
					weight_update_n(i, j) = dSnp1_dWn(i, j) * dTotalError_dSnp1(0, j) * speed;
				}
		}

		//Weight update of the last neuron layer.
		static void Compute_final_weight_update(Matrix<T> &last_weight_update, const Matrix<T> &error_vector, const Matrix<T> &last_dSnp1_dWn, T speed) {
			assert(last_weight_update.getDimensions() == last_dSnp1_dWn.getDimensions());
			for (size_t i = 0; i < last_weight_update.getRowCount(); i++)
				for (size_t j = 0; j < last_weight_update.getColumnCount(); j++) {
					last_weight_update(i, j) = last_dSnp1_dWn(i, j) * error_vector(0, j) * speed;
				}
		}

		//returns output(0,i) := d(signal_layer_next(0,i))/d(biases(0,j))
		static void Compute_dSnp1_dBn(Matrix<T> &output, const Matrix<T> &signal_layer_next, Neuron_Type neuron_type) {
			assert(output.getDimensions() == signal_layer_next.getDimensions());
			for (auto i = 0; i < output.getColumnCount(); ++i) {
				auto dSnp1_dNn = evalute_perceptron_prime(signal_layer_next(0, i), neuron_type);
				output(0, i) = dSnp1_dNn; //because dNn/dBn = 1
			}
		}

		//returns the bias updates of the Nth layer
		static void Compute_bias_update(Matrix<T> &bias_update, const Matrix<T> &dTotalError_dSnp1, const Matrix<T> &dSnp1_dBn, T speed) {
			for (size_t idx = 0; idx < bias_update.getColumnCount(); ++idx) {
				bias_update(0, idx) = dTotalError_dSnp1(0, idx) * dSnp1_dBn(0, idx) * speed;
			}
		}
		
		//Bias update for the last neuron layer
		static void Compute_final_bias_update(Matrix<T> &last_bias_update, const Matrix<T> &error_vector, const Matrix<T> &last_dSnp1_dBn, T speed) {
			for (size_t idx = 0; idx < last_bias_update.getColumnCount(); ++idx) {
				last_bias_update(0, idx) = last_dSnp1_dBn(0, idx) * error_vector(0, idx) * speed;
			}
		}

	private:
		/*
		Back propagation variables:
		*/
			Matrix<T> error_vector;

			//dSnp2_dSnp1[n](i,j) := d(signal_nodes[n+2](0,i))/d(signal_nodes[n+1](0,j))
			std::vector<Matrix<T>> dSnp2_dSnp1;

			//dSOutput_dSnp1[n](i,j) := d(signal_nodes[last](0,i))/d(signal_nodes[n+1](0,j))
			std::vector<Matrix<T>> dSOutput_dSnp1;

			//dError_dSnp1[n](i,j) := d(error(0,i))/d(signal_nodes[n+1](0,j))
			std::vector<Matrix<T>> dError_dSnp1;

			//dTotalError_dSnp1[n](0,j) := d(TotalError)/d(signal_nodes[n+1](0,j)), thios is equal to sum of every row in dError_dSnp1.
			std::vector<Matrix<T>> dTotalError_dSnp1;

			//dSnp1_dWn[n](j,i) := d(signal_nodes[n+1](0,i))/d(weights[n](j,i))
			std::vector<Matrix<T>> dSnp1_dWn;

			//dSnp1_dBn[n](i,j) := d(signal_nodes[n+1](0,i))/d(biases[n](0,j))
			std::vector<Matrix<T>> dSnp1_dBn;

			//Resultant updates
			std::vector<Matrix<T>> weight_updates;
			std::vector<Matrix<T>> bias_updates;
		
		/*
		Forward propagation variables:
		*/
			//Every signal_nodes[i], i!=0, represents a layer of neuron output values. signal_nodes[0] := input. All signal_nodes are row vectors
			std::vector<Matrix<T>> signal_nodes;

			//weights[m](i,j) := Weight from signal_nodes[m](0,i) to network_nodes[m](0,j). 
			std::vector<Matrix<T>> weights;

			//network_nodes[i] := signal_nodes[i] * weights[i]. All network_nodes are row vectors
			std::vector<Matrix<T>> network_nodes;

			//biases[i] := bias values on the neurons
			std::vector<Matrix<T>> biases;

			std::vector<Neuron_Type> neuron_types;

			std::vector<T> speeds;
	
	public:

		const std::vector<Matrix<T>> &getWeights() const{
			return weights;
		};

		const std::vector<Matrix<T>> &getBiases() const{
			return biases;
		};

	};
}