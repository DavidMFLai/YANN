#pragma once
#include "ReferenceMatrix.h"
#include <cmath>
#include <cassert>
#include <random>
#include <string>

using std::string;
namespace CPPANN {
	enum class Neuron_Type {
		Sigmoid,
		Tanh
	};

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

		ANNBuilder &set_weights(size_t starting_layer_index, std::initializer_list<std::initializer_list<T>> matrix_initializer_list) {
			if (weight_matrices.size() <= starting_layer_index) {
				weight_matrices.resize(starting_layer_index + 1);
			}
			weight_matrices.at(starting_layer_index) = ReferenceMatrix<T> { matrix_initializer_list };
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

		const std::vector<ReferenceMatrix<T>> &get_weight_matrices() const {
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

		static bool Neuron_count_weights_count_mismatch(const std::vector<size_t> &neuron_counts, const std::vector<ReferenceMatrix<T>> &weight_matrices, size_t layer_idx) {
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

		static void Make_random_weights_if_needed(std::vector<ReferenceMatrix<T>> &weight_matrices, const std::vector<size_t> &neuron_counts, const Random_number_generator<T> &random_number_generator) {
			if (weight_matrices.size() != neuron_counts.size() - 1) {
				weight_matrices.resize(neuron_counts.size() - 1);
			}
			for (size_t idx = 0; idx < weight_matrices.size(); idx++) {
				if (Neuron_count_weights_count_mismatch(neuron_counts, weight_matrices, idx)) {
					//create a new Matrix
					ReferenceMatrix<T> random_matrix { neuron_counts.at(idx), neuron_counts.at(idx+1) };
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
		std::vector<ReferenceMatrix<T>> weight_matrices;
	};

	template<typename T>
	class ANN {
	private:
		friend ANN<T> ANNBuilder<T>::build();
		ANN(const std::vector<size_t> &signal_counts, 
			const std::vector<std::vector<T>> &biases_of_each_layer, 
			const std::vector<ReferenceMatrix<T>> &weight_matrices, 
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
					signal_nodes.at(idx) = ReferenceMatrix<T>{1, signal_counts[idx]};
				}
				else {
					signal_nodes.at(idx) = ReferenceMatrix<T>{1, signal_counts[idx]};
					network_nodes.at(idx - 1) = ReferenceMatrix<T>{1, signal_counts[idx]};
					biases.at(idx - 1) = biases_of_each_layer.at(idx - 1);
					weights.at(idx - 1) = weight_matrices.at(idx - 1);
				}
			}

			//Create additional matrices required by backward propagation
			dSnp1_dNn.resize(layers_count - 1);
			dSnp1_dBn.resize(layers_count - 1);
			dSnp1_dWn.resize(layers_count - 1);
			weight_updates.resize(layers_count - 1);
			bias_updates.resize(layers_count - 1);
			dSOutput_dSnp1.resize(layers_count - 2);
			dError_dSnp1.resize(layers_count - 2);
			dSnp2_dSnp1.resize(layers_count - 2);
			dTotalError_dSnp1.resize(layers_count - 2);
			error_vector = ReferenceMatrix<T>{ signal_nodes.back().getRowCount(), signal_nodes.back().getColumnCount() };

			for (size_t idx = 0; idx < layers_count - 1; idx++) {
				if (idx == 0) {
					dSnp1_dNn.at(idx) = ReferenceMatrix<T>{ 1, network_nodes.at(idx).getColumnCount() };
					dSnp1_dBn.at(idx) = ReferenceMatrix<T>{ 1, network_nodes.at(idx).getColumnCount() };
					dSnp1_dWn.at(idx) = ReferenceMatrix<T>{ weights.at(idx).getRowCount(),  weights.at(idx).getColumnCount() };
					weight_updates.at(idx) = ReferenceMatrix<T>{ weights.at(idx).getRowCount(), weights.at(idx).getColumnCount() };
					bias_updates.at(idx) = ReferenceMatrix<T>{ biases.at(idx).getRowCount(), biases.at(idx).getColumnCount() };
				}
				else {
					dSnp1_dNn.at(idx) = ReferenceMatrix<T>{ 1, network_nodes.at(idx).getColumnCount() };
					dSnp1_dBn.at(idx) = ReferenceMatrix<T>{ 1, network_nodes.at(idx).getColumnCount() };
					dSnp1_dWn.at(idx) = ReferenceMatrix<T>{ weights.at(idx).getRowCount(),  weights.at(idx).getColumnCount() };
					weight_updates.at(idx) = ReferenceMatrix<T>{ weights.at(idx).getRowCount(), weights.at(idx).getColumnCount() };
					bias_updates.at(idx) = ReferenceMatrix<T>{ biases.at(idx).getRowCount(), biases.at(idx).getColumnCount() };
					dSOutput_dSnp1.at(idx - 1) = ReferenceMatrix<T>{signal_nodes.back().getColumnCount(), signal_nodes.at(idx).getColumnCount()};
					dError_dSnp1.at(idx - 1) = ReferenceMatrix<T>{ signal_nodes.back().getColumnCount(), signal_nodes.at(idx).getColumnCount() };
					dTotalError_dSnp1.at(idx - 1) = ReferenceMatrix<T>{ 1, signal_nodes.at(idx).getColumnCount() };
				}
			}

			for (size_t idx = 1; idx < layers_count - 2; idx++) {
				dSnp2_dSnp1.at(idx - 1) = ReferenceMatrix<T>{ signal_nodes.at(idx + 1).getColumnCount(), signal_nodes.at(idx).getColumnCount() };
			}
		}
	public:
		const std::vector<T> &forward_propagate(const std::vector<T> &input) {
			ReferenceMatrix<T>::Copy_from_vector(signal_nodes[0], input);
			for (int i = 0; i < weights.size(); i++) {
				ReferenceMatrix<T>::Multiply(network_nodes[i], signal_nodes[i], weights[i]);
				ReferenceMatrix<T>::Add(network_nodes[i], network_nodes[i], biases[i]);
				if (neuron_types[i] == Neuron_Type::Sigmoid) {
					ReferenceMatrix<T>::Per_Element_Sigmoid(signal_nodes[i + 1], network_nodes[i]);
				}
				else {
					ReferenceMatrix<T>::Per_Element_Tanh(signal_nodes[i + 1], network_nodes[i]);
				}
			}
			return signal_nodes.back().getElems();
		}

		void back_propagate(const std::vector<T> &expected_as_vector) {
			
			ReferenceMatrix<T> expected{ expected_as_vector };
			
			//compute error
			ReferenceMatrix<T>::Minus(error_vector, signal_nodes.back(), expected);

			//compute dSnp1_dNn
			for (size_t idx = 0; idx < dSnp1_dNn.size(); ++idx) {
				Compute_dSnp1_dNn(dSnp1_dNn[idx], signal_nodes[idx + 1], neuron_types[idx]);
			}

			//compute dSnp2_dSnp1
			for (size_t idx = 0; idx < dSnp2_dSnp1.size() - 1; ++idx) {
				Compute_dSnp1_dSn(dSnp2_dSnp1[idx], dSnp1_dNn[idx + 1], weights[idx + 1]);
			}

			//compute dSOutput_dSnp1
			Compute_dSnp1_dSn(dSOutput_dSnp1.back(), dSnp1_dNn.back(), weights.back());
			if (dSOutput_dSnp1.size() >= 2) {
				for (size_t idx = dSOutput_dSnp1.size() - 2; idx != std::numeric_limits<size_t>::max(); --idx) {
					ReferenceMatrix<T>::Multiply(dSOutput_dSnp1[idx], dSOutput_dSnp1[idx + 1], dSnp2_dSnp1[idx]);
				}
			}

			//compute dSnp1_dWn
			for (auto idx = 0; idx < dSnp1_dWn.size(); ++idx) {
				Compute_dSnp1_dWn(dSnp1_dWn[idx], dSnp1_dNn[idx], signal_nodes[idx]);
			}

			//compute dSError_dSnp1
			for (size_t idx = 0; idx < dError_dSnp1.size(); ++idx) {
				ReferenceMatrix<T>::Per_Row_Multiply(dError_dSnp1[idx], error_vector, dSOutput_dSnp1[idx]);
			}

			// compute weight updates and apply them
			Compute_dTotalError_dSnp1(dTotalError_dSnp1, dError_dSnp1);
			for (size_t idx = 0; idx < weight_updates.size()-1; ++idx) {
				Compute_weight_update(weight_updates[idx], dTotalError_dSnp1[idx], dSnp1_dWn[idx], speeds[idx]);
				ReferenceMatrix<T>::subtract_andThen_assign(weights[idx], weight_updates[idx]);
			}
			Compute_final_weight_update(weight_updates.back(), error_vector, dSnp1_dWn.back(), speeds.back());
			ReferenceMatrix<T>::subtract_andThen_assign(weights.back(), weight_updates.back());

			//compute dSnp1_dBn		
			for (size_t idx = 0; idx < dSnp1_dBn.size(); ++idx) {
				Compute_dSnp1_dBn(dSnp1_dBn[idx], dSnp1_dNn[idx]);
			}

			//compute bias updates and apply them
			for (size_t idx = 0; idx < bias_updates.size() - 1; ++idx) {
				Compute_bias_update(bias_updates[idx], dTotalError_dSnp1[idx], dSnp1_dBn[idx], speeds[idx]);
				ReferenceMatrix<T>::subtract_andThen_assign(biases[idx], bias_updates[idx]);
			}
			Compute_final_bias_update(bias_updates.back(), error_vector, dSnp1_dBn.back(), speeds.back());
			ReferenceMatrix<T>::subtract_andThen_assign(biases.back(), bias_updates.back());
		}

	private:
		//retuns output(0, i) = d(signal_layer_next(0, i)/network_layer(0, i)
		static void Compute_dSnp1_dNn(ReferenceMatrix<T> &output, const ReferenceMatrix<T> &signal_layer_next, Neuron_Type neuron_type){
			assert(output.getColumnCount() == signal_layer_next.getColumnCount());
			if (neuron_type == Neuron_Type::Sigmoid) {
				ReferenceMatrix<T>::Per_Element_Sigmoid_Prime(output, signal_layer_next);
			}
			else {
				ReferenceMatrix<T>::Per_Element_Tanh_Prime(output, signal_layer_next);
			}
		}

		static void Compute_dSnp1_dSn(ReferenceMatrix<T> &output, const ReferenceMatrix<T> &dSnp1_dNn, const ReferenceMatrix<T> &weights) {
			assert(output.getRowCount() == dSnp1_dNn.getColumnCount());
			assert(output.getRowCount() == weights.getColumnCount());
			assert(output.getColumnCount() == weights.getRowCount());
			ReferenceMatrix<T>::Per_Column_Multiply_AndThen_Transpose(output, dSnp1_dNn, weights);
		}

		//returns dTotalError_dSnp1(0,j) := d(sum(error_vector))/d(Snp1(0,j)) 
		static void Compute_dTotalError_dSnp1(std::vector<ReferenceMatrix<T>> &dTotalError_dSnp1, const std::vector<ReferenceMatrix<T>> &dSOutput_dSnp1) {
			assert(dTotalError_dSnp1.size() == dSOutput_dSnp1.size());
			for (size_t idx = 0; idx < dTotalError_dSnp1.size(); ++idx) {
				ReferenceMatrix<T>::Sum_of_rows(dTotalError_dSnp1[idx], dSOutput_dSnp1[idx]);
			}
		}

		//returns output(i,j) := d(signal_layer_next(0,j))/d(weights(i,j))
		static void Compute_dSnp1_dWn(ReferenceMatrix<T> &output, const ReferenceMatrix<T> &dSnp1_dNn, const ReferenceMatrix<T> &signal_layer) {
			assert(output.getRowCount() == signal_layer.getColumnCount());
			assert(output.getColumnCount() == dSnp1_dNn.getColumnCount());
			ReferenceMatrix<T>::Outer_product(output, signal_layer, dSnp1_dNn);
		}

		//returns the weight_update of the Nth layer
		static void Compute_weight_update(ReferenceMatrix<T> &weight_update_n, const ReferenceMatrix<T> &dTotalError_dSnp1, const ReferenceMatrix<T> &dSnp1_dWn, T speed) {
			assert(weight_update_n.getDimensions() == dSnp1_dWn.getDimensions());
			assert(weight_update_n.getColumnCount() == dSnp1_dWn.getColumnCount());
			ReferenceMatrix<T>::Per_Column_Multiply_AndThen_Scale(weight_update_n, dTotalError_dSnp1, dSnp1_dWn, speed);
		}

		//Weight update of the last neuron layer.
		static void Compute_final_weight_update(ReferenceMatrix<T> &last_weight_update, const ReferenceMatrix<T> &error_vector, const ReferenceMatrix<T> &last_dSnp1_dWn, T speed) {
			assert(last_weight_update.getDimensions() == last_dSnp1_dWn.getDimensions());
			ReferenceMatrix<T>::Per_Column_Multiply_AndThen_Scale(last_weight_update, error_vector, last_dSnp1_dWn, speed);
		}

		//returns output(0,i) := d(signal_layer_next(0,i))/d(biases(0,j))
		static void Compute_dSnp1_dBn(ReferenceMatrix<T> &output, const ReferenceMatrix<T> &dSnp1_dNn) {
			assert(output.getDimensions() == dSnp1_dNn.getDimensions());
			ReferenceMatrix<T>::copy(output, dSnp1_dNn);
		}

		//returns the bias updates of the Nth layer
		static void Compute_bias_update(ReferenceMatrix<T> &bias_update, const ReferenceMatrix<T> &dTotalError_dSnp1, const ReferenceMatrix<T> &dSnp1_dBn, T speed) {
			ReferenceMatrix<T>::Row_Vectors_Per_Element_Multiply_AndThen_Scale(bias_update, dTotalError_dSnp1, dSnp1_dBn, speed);
		}
		
		//Bias update for the last neuron layer
		static void Compute_final_bias_update(ReferenceMatrix<T> &last_bias_update, const ReferenceMatrix<T> &error_vector, const ReferenceMatrix<T> &last_dSnp1_dBn, T speed) {
			ReferenceMatrix<T>::Row_Vectors_Per_Element_Multiply_AndThen_Scale(last_bias_update, error_vector, last_dSnp1_dBn, speed);
		}

	private:
		/*
		Back propagation variables:
		*/
			ReferenceMatrix<T> error_vector;

			//dSnp1_dNn[n](0,i) := d(signal_nodes[n+1](0,i))/d(network_nodes[n](0,i))
			std::vector<ReferenceMatrix<T>> dSnp1_dNn;

			//dSnp2_dSnp1[n](i,j) := d(signal_nodes[n+2](0,i))/d(signal_nodes[n+1](0,j))
			std::vector<ReferenceMatrix<T>> dSnp2_dSnp1;

			//dSOutput_dSnp1[n](i,j) := d(signal_nodes[last](0,i))/d(signal_nodes[n+1](0,j))
			std::vector<ReferenceMatrix<T>> dSOutput_dSnp1;

			//dError_dSnp1[n](i,j) := d(error(0,i))/d(signal_nodes[n+1](0,j))
			std::vector<ReferenceMatrix<T>> dError_dSnp1;

			//dTotalError_dSnp1[n](0,j) := d(TotalError)/d(signal_nodes[n+1](0,j)), thios is equal to sum of every row in dError_dSnp1.
			std::vector<ReferenceMatrix<T>> dTotalError_dSnp1;

			//dSnp1_dWn[n](j,i) := d(signal_nodes[n+1](0,i))/d(weights[n](j,i))
			std::vector<ReferenceMatrix<T>> dSnp1_dWn;

			//dSnp1_dBn[n](i,j) := d(signal_nodes[n+1](0,i))/d(biases[n](0,j))
			std::vector<ReferenceMatrix<T>> dSnp1_dBn;

			//Resultant updates
			std::vector<ReferenceMatrix<T>> weight_updates;
			std::vector<ReferenceMatrix<T>> bias_updates;
		
		/*
		Forward propagation variables:
		*/
			//Every signal_nodes[i], i!=0, represents a layer of neuron output values. signal_nodes[0] := input. All signal_nodes are row vectors
			std::vector<ReferenceMatrix<T>> signal_nodes;

			//weights[m](i,j) := Weight from signal_nodes[m](0,i) to network_nodes[m](0,j). 
			std::vector<ReferenceMatrix<T>> weights;

			//network_nodes[i] := signal_nodes[i] * weights[i]. All network_nodes are row vectors
			std::vector<ReferenceMatrix<T>> network_nodes;

			//biases[i] := bias values on the neurons
			std::vector<ReferenceMatrix<T>> biases;

			std::vector<Neuron_Type> neuron_types;

			std::vector<T> speeds;
	
	public:

		const std::vector<ReferenceMatrix<T>> &getWeights() const{
			return weights;
		};

		const std::vector<ReferenceMatrix<T>> &getBiases() const{
			return biases;
		};

	};
}