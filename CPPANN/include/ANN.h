#pragma once
#include "ReferenceMatrix.h"
#include "ReferenceMatrixBuilder.h"
#include <cmath>
#include <cassert>
#include <random>
#include <string>
#include <map>
#include <memory>

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

		std::vector<T> generate_vector(size_t size) const{
			std::vector<T> retval(size);
			for (auto &val : retval) {
				val = dist(gen);
			}
			return retval;
		}

	private:
		std::random_device rd;
		mutable std::mt19937 gen;
		mutable std::normal_distribution<T> dist;
	};

	template<typename T>
	class ANNBuilder {
	public:
		ANNBuilder()
			: output_neuron_count{ 0 }, matrix_builder{nullptr}
		{}

		ANNBuilder &set_input_layer(uint64_t size) {
			layer_idx_to_neuron_count.insert({ 0, size });
			return *this;
		}

		//Hidden layer index is zero based
		ANNBuilder &set_hidden_layer(size_t hidden_layer_index, Neuron_Type type, T speed, uint64_t size) {
			//setup neuron counts
			//0th hidden layer is the 1st layer in the ANN. (0th layer in the ANN is the input layer)
			size_t layer_index = hidden_layer_index + 1;
			layer_idx_to_neuron_count.insert({ layer_index, size });

			//setup neuron_types
			layer_idx_to_neuron_type.insert({ hidden_layer_index, type });

			//setup speeds
			layer_idx_to_speeds.insert({ hidden_layer_index , speed });
			return *this;
		}

		ANNBuilder &set_output_layer(Neuron_Type type, T speed, uint64_t size) {
			output_neuron_type = type;
			output_speed = speed;
			output_neuron_count = size;
			return *this;
		}

		ANNBuilder &set_bias(size_t layer_index, const std::vector<T> &input) {
			layer_idx_to_biases.insert({ layer_index, input });
			return *this;
		}

		ANNBuilder &set_weights(size_t layer_index, std::initializer_list<std::initializer_list<T>> matrix_initializer_lists) {
			//construct vector of vectors from matrix_initializer_lists
			std::vector<std::vector<T>> matrix_values;
			for (auto matrix_initializer_list : matrix_initializer_lists) {
				matrix_values.push_back(std::vector<T>{matrix_initializer_list});
			}

			layer_idx_to_weight_matrix_values.insert({ layer_index, matrix_values });
			return *this;
		}

	private:
		static std::vector<uint64_t> Create_neuron_counts(const std::map<size_t, uint64_t> &layer_idx_to_neuron_count, size_t output_neuron_count) {
			std::vector<uint64_t> neuron_counts;
			for (auto layer_idx_neuron_count_pair : layer_idx_to_neuron_count) {
				neuron_counts.push_back(layer_idx_neuron_count_pair.second);
			}
			neuron_counts.push_back(output_neuron_count);
			return neuron_counts;
		}

		static std::vector<Neuron_Type> Create_neuron_types(const std::map<size_t, Neuron_Type> &layer_idx_to_neuron_type, Neuron_Type output_neuron_type) {
			std::vector<Neuron_Type> neuron_types;
			for (auto layer_idx_neuron_type_pair : layer_idx_to_neuron_type) {
				neuron_types.push_back(layer_idx_neuron_type_pair.second);
			}
			neuron_types.push_back(output_neuron_type);
			return neuron_types;
		}

		static std::vector<T> Create_speeds(const std::map<size_t, T> &layer_idx_to_speeds, T output_speed) {
			std::vector<T> speeds;
			for (auto layer_idx_speed_pair : layer_idx_to_speeds) {
				speeds.push_back(layer_idx_speed_pair.second);
			}
			speeds.push_back(output_speed);
			return speeds;
		}

		static std::vector<std::vector<T>> Create_biases(const std::map<size_t, std::vector<T>> &layer_idx_to_biases, const Random_number_generator<T> &gen, std::vector<uint64_t> neuron_counts) {
			std::vector<std::vector<T>> biases(neuron_counts.size() - 1);
			for (size_t idx = 0; idx < biases.size(); idx++) {
				//create a new random bias vector of length == neuron_counts.at(idx + 1)
				biases.at(idx) = gen.generate_vector(neuron_counts.at(idx + 1));
			}
			//overwrite the random values with values from the settings
			for (auto layer_idx_biases_pair : layer_idx_to_biases) {
				biases.at(layer_idx_biases_pair.first) = layer_idx_biases_pair.second;
			}
			return biases;
		}

		static std::vector<std::unique_ptr<Matrix<T>>> Create_weight_matrices(const std::map<size_t, std::vector<std::vector<T>>> &layer_idx_to_weight_matrix_values, const Random_number_generator<T> &gen, std::vector<uint64_t> neuron_counts, MatrixBuilder<T> *matrix_builder) {
			std::vector<std::unique_ptr<Matrix<T>>> weight_matrices(neuron_counts.size() - 1);
			for (size_t idx = 0; idx < weight_matrices.size(); idx++) {
				//create a new random bias vector of length == neuron_counts.at(idx + 1)
				auto random_matrix = matrix_builder->create(neuron_counts.at(idx), neuron_counts.at(idx + 1));
				for (size_t i = 0; i < neuron_counts.at(idx); i++) {
					for (size_t j = 0; j < neuron_counts.at(idx + 1); j++) {
						random_matrix->at(i, j) = gen.generate();
					}
				}
				weight_matrices.at(idx) = std::move(random_matrix);
			}
			//overwrite the random values with values from the settings
			for (auto layer_idx_weight_matrix_pair : layer_idx_to_weight_matrix_values) {
				auto random_matrix = matrix_builder->create(layer_idx_weight_matrix_pair.second);
				weight_matrices.at(layer_idx_weight_matrix_pair.first) = std::move(random_matrix);
			}
			return weight_matrices;
		}

	public:

		ANN<T> build() {
			//todo: check if the user has set the ANN properly
			if (matrix_builder.get() == nullptr) {
				matrix_builder.reset(new ReferenceMatrixBuilder<T>);
			}
			
			auto neuron_counts = Create_neuron_counts(this->layer_idx_to_neuron_count, this->output_neuron_count);
			auto biases = Create_biases(this->layer_idx_to_biases, this->random_number_generator, neuron_counts);
			auto weight_matrices = Create_weight_matrices(this->layer_idx_to_weight_matrix_values, this->random_number_generator, neuron_counts, this->matrix_builder.get());
			auto neuron_types = Create_neuron_types(this->layer_idx_to_neuron_type, this->output_neuron_type);
			auto speeds = Create_speeds(this->layer_idx_to_speeds, output_speed);

			ANN<T> retval{ neuron_counts, biases, weight_matrices, neuron_types, speeds, this->matrix_builder.get() };
			return retval;
		}

	private:
		/*
		Input ANN settings
		*/
		size_t output_neuron_count;
		Neuron_Type output_neuron_type;
		T output_speed;
		std::map<size_t, std::vector<T>> layer_idx_to_biases;
		std::map<size_t, std::vector<std::vector<T>>> layer_idx_to_weight_matrix_values; //if settings_weight_matrices_value.at(0)[1][2] == 0th Weight Matrix's value at (1,2);
		std::map<size_t, uint64_t> layer_idx_to_neuron_count;
		std::map<size_t, Neuron_Type> layer_idx_to_neuron_type;
		std::map<size_t, T> layer_idx_to_speeds;

		/*
		Others
		*/		
		Random_number_generator<T> random_number_generator;
		std::unique_ptr<MatrixBuilder<T>> matrix_builder;
	};

	template<typename T>
	class ANN {
	private:
		friend ANN<T> ANNBuilder<T>::build();
		ANN(const std::vector<size_t> &signal_counts, 
			const std::vector<std::vector<T>> &settings_biases, 
			std::vector<std::unique_ptr<Matrix<T>>> &weight_matrices, 
			const std::vector<Neuron_Type> &neuron_types,
			const std::vector<T> &speeds,
			MatrixBuilder<T> *matrix_builder
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
					signal_nodes.at(idx) = matrix_builder->create(1, signal_counts[idx]);
				}
				else {
					signal_nodes.at(idx) = matrix_builder->create(1, signal_counts[idx]);
					network_nodes.at(idx - 1) = matrix_builder->create(1, signal_counts[idx]);
					biases.at(idx - 1) = matrix_builder->create(settings_biases.at(idx - 1));
					weights.at(idx - 1) = std::move(weight_matrices.at(idx - 1));
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
			error_vector = matrix_builder->create( signal_nodes.back()->getRowCount(), signal_nodes.back()->getColumnCount() );
			expected_vector = matrix_builder->create(signal_nodes.back()->getRowCount(), signal_nodes.back()->getColumnCount());

			for (size_t idx = 0; idx < layers_count - 1; idx++) {
				if (idx == 0) {
					dSnp1_dNn.at(idx) = matrix_builder->create(1, network_nodes.at(idx)->getColumnCount());
					dSnp1_dBn.at(idx) = matrix_builder->create( 1, network_nodes.at(idx)->getColumnCount() );
					dSnp1_dWn.at(idx) = matrix_builder->create( weights.at(idx)->getRowCount(),  weights.at(idx)->getColumnCount() );
					weight_updates.at(idx) = matrix_builder->create( weights.at(idx)->getRowCount(), weights.at(idx)->getColumnCount() );
					bias_updates.at(idx) = matrix_builder->create( biases.at(idx)->getRowCount(), biases.at(idx)->getColumnCount() );
				}
				else {
					dSnp1_dNn.at(idx) = matrix_builder->create( 1, network_nodes.at(idx)->getColumnCount() );
					dSnp1_dBn.at(idx) = matrix_builder->create( 1, network_nodes.at(idx)->getColumnCount() );
					dSnp1_dWn.at(idx) = matrix_builder->create( weights.at(idx)->getRowCount(),  weights.at(idx)->getColumnCount() );
					weight_updates.at(idx) = matrix_builder->create( weights.at(idx)->getRowCount(), weights.at(idx)->getColumnCount() );
					bias_updates.at(idx) = matrix_builder->create( biases.at(idx)->getRowCount(), biases.at(idx)->getColumnCount() );
					dSOutput_dSnp1.at(idx - 1) = matrix_builder->create(signal_nodes.back()->getColumnCount(), signal_nodes.at(idx)->getColumnCount());
					dError_dSnp1.at(idx - 1) = matrix_builder->create( signal_nodes.back()->getColumnCount(), signal_nodes.at(idx)->getColumnCount() );
					dTotalError_dSnp1.at(idx - 1) = matrix_builder->create( 1, signal_nodes.at(idx)->getColumnCount() );
				}
			}

			for (size_t idx = 1; idx < layers_count - 2; idx++) {
				dSnp2_dSnp1.at(idx - 1) = matrix_builder->create( signal_nodes.at(idx + 1)->getColumnCount(), signal_nodes.at(idx)->getColumnCount() );
			}
		}
	public:
		const std::vector<T> &forward_propagate(const std::vector<T> &input) {
			Matrix<T>::Copy_from_vector(*signal_nodes[0], input);
			for (int i = 0; i < weights.size(); i++) {
				Matrix<T>::Multiply(*network_nodes[i], *signal_nodes[i], *weights[i]);
				Matrix<T>::Add(*network_nodes[i], *network_nodes[i], *biases[i]);
				if (neuron_types[i] == Neuron_Type::Sigmoid) {
					Matrix<T>::Per_Element_Sigmoid(*signal_nodes[i + 1], *network_nodes[i]);
				}
				else {
					Matrix<T>::Per_Element_Tanh(*signal_nodes[i + 1], *network_nodes[i]);
				}
			}
			return signal_nodes.back()->getElems();
		}

		void back_propagate(const std::vector<T> &expected_as_vector) {
			Matrix<T>::Copy_from_vector(*expected_vector, expected_as_vector);

			//compute error
			Matrix<T>::Minus(*error_vector, *signal_nodes.back(), *expected_vector);

			//compute dSnp1_dNn
			for (size_t idx = 0; idx < dSnp1_dNn.size(); ++idx) {
				Compute_dSnp1_dNn(*dSnp1_dNn[idx], *signal_nodes[idx + 1], neuron_types[idx]);
			}

			//compute dSnp2_dSnp1
			for (size_t idx = 0; idx < dSnp2_dSnp1.size() - 1; ++idx) {
				Compute_dSnp1_dSn(*dSnp2_dSnp1[idx], *dSnp1_dNn[idx + 1], *weights[idx + 1]);
			}

			//compute dSOutput_dSnp1
			Compute_dSnp1_dSn(*dSOutput_dSnp1.back(), *dSnp1_dNn.back(), *weights.back());
			if (dSOutput_dSnp1.size() >= 2) {
				for (size_t idx = dSOutput_dSnp1.size() - 2; idx != std::numeric_limits<size_t>::max(); --idx) {
					Matrix<T>::Multiply(*dSOutput_dSnp1[idx], *dSOutput_dSnp1[idx + 1], *dSnp2_dSnp1[idx]);
				}
			}

			//compute dSnp1_dWn
			for (auto idx = 0; idx < dSnp1_dWn.size(); ++idx) {
				Compute_dSnp1_dWn(*dSnp1_dWn[idx], *dSnp1_dNn[idx], *signal_nodes[idx]);
			}

			//compute dSError_dSnp1
			for (size_t idx = 0; idx < dError_dSnp1.size(); ++idx) {
				Matrix<T>::Per_Row_Multiply(*dError_dSnp1[idx], *error_vector, *dSOutput_dSnp1[idx]);
			}

			// compute weight updates and apply them
			Compute_dTotalError_dSnp1(dTotalError_dSnp1, dError_dSnp1);
			for (size_t idx = 0; idx < weight_updates.size()-1; ++idx) {
				Compute_weight_update(*weight_updates[idx], *dTotalError_dSnp1[idx], *dSnp1_dWn[idx], speeds[idx]);
				Matrix<T>::subtract_andThen_assign(*weights[idx], *weight_updates[idx]);
			}
			Compute_final_weight_update(*weight_updates.back(), *error_vector, *dSnp1_dWn.back(), speeds.back());
			Matrix<T>::subtract_andThen_assign(*weights.back(), *weight_updates.back());

			//compute dSnp1_dBn		
			for (size_t idx = 0; idx < dSnp1_dBn.size(); ++idx) {
				Compute_dSnp1_dBn(*dSnp1_dBn[idx], *dSnp1_dNn[idx]);
			}

			//compute bias updates and apply them
			for (size_t idx = 0; idx < bias_updates.size() - 1; ++idx) {
				Compute_bias_update(*bias_updates[idx], *dTotalError_dSnp1[idx], *dSnp1_dBn[idx], speeds[idx]);
				Matrix<T>::subtract_andThen_assign(*biases[idx], *bias_updates[idx]);
			}
			Compute_final_bias_update(*bias_updates.back(), *error_vector, *dSnp1_dBn.back(), speeds.back());
			Matrix<T>::subtract_andThen_assign(*biases.back(), *bias_updates.back());
		}

	private:
		//retuns output(0, i) = d(signal_layer_next(0, i)/network_layer(0, i)
		static void Compute_dSnp1_dNn(Matrix<T> &output, const Matrix<T> &signal_layer_next, Neuron_Type neuron_type){
			assert(output.getColumnCount() == signal_layer_next.getColumnCount());
			if (neuron_type == Neuron_Type::Sigmoid) {
				Matrix<T>::Per_Element_Sigmoid_Prime(output, signal_layer_next);
			}
			else {
				Matrix<T>::Per_Element_Tanh_Prime(output, signal_layer_next);
			}
		}

		static void Compute_dSnp1_dSn(Matrix<T> &output, const Matrix<T> &dSnp1_dNn, const Matrix<T> &weights) {
			assert(output.getRowCount() == dSnp1_dNn.getColumnCount());
			assert(output.getRowCount() == weights.getColumnCount());
			assert(output.getColumnCount() == weights.getRowCount());
			Matrix<T>::Per_Column_Multiply_AndThen_Transpose(output, dSnp1_dNn, weights);
		}

		//returns dTotalError_dSnp1(0,j) := d(sum(error_vector))/d(Snp1(0,j)) 
		static void Compute_dTotalError_dSnp1(std::vector<std::unique_ptr<Matrix<T>>> &dTotalError_dSnp1, const std::vector<std::unique_ptr<Matrix<T>>> &dSOutput_dSnp1) {
			assert(dTotalError_dSnp1.size() == dSOutput_dSnp1.size());
			for (size_t idx = 0; idx < dTotalError_dSnp1.size(); ++idx) {
				Matrix<T>::Sum_of_rows(*dTotalError_dSnp1[idx], *dSOutput_dSnp1[idx]);
			}
		}

		//returns output(i,j) := d(signal_layer_next(0,j))/d(weights(i,j))
		static void Compute_dSnp1_dWn(Matrix<T> &output, const Matrix<T> &dSnp1_dNn, const Matrix<T> &signal_layer) {
			assert(output.getRowCount() == signal_layer.getColumnCount());
			assert(output.getColumnCount() == dSnp1_dNn.getColumnCount());
			Matrix<T>::Outer_product(output, signal_layer, dSnp1_dNn);
		}

		//returns the weight_update of the Nth layer
		static void Compute_weight_update(Matrix<T> &weight_update_n, const Matrix<T> &dTotalError_dSnp1, const Matrix<T> &dSnp1_dWn, T speed) {
			assert(weight_update_n.getDimensions() == dSnp1_dWn.getDimensions());
			assert(weight_update_n.getColumnCount() == dSnp1_dWn.getColumnCount());
			Matrix<T>::Per_Column_Multiply_AndThen_Scale(weight_update_n, dTotalError_dSnp1, dSnp1_dWn, speed);
		}

		//Weight update of the last neuron layer.
		static void Compute_final_weight_update(Matrix<T> &last_weight_update, const Matrix<T> &error_vector, const Matrix<T> &last_dSnp1_dWn, T speed) {
			assert(last_weight_update.getDimensions() == last_dSnp1_dWn.getDimensions());
			Matrix<T>::Per_Column_Multiply_AndThen_Scale(last_weight_update, error_vector, last_dSnp1_dWn, speed);
		}

		//returns output(0,i) := d(signal_layer_next(0,i))/d(biases(0,j))
		static void Compute_dSnp1_dBn(Matrix<T> &output, const Matrix<T> &dSnp1_dNn) {
			assert(output.getDimensions() == dSnp1_dNn.getDimensions());
			Matrix<T>::Copy(output, dSnp1_dNn);
		}

		//returns the bias updates of the Nth layer
		static void Compute_bias_update(Matrix<T> &bias_update, const Matrix<T> &dTotalError_dSnp1, const Matrix<T> &dSnp1_dBn, T speed) {
			Matrix<T>::Row_Vectors_Per_Element_Multiply_AndThen_Scale(bias_update, dTotalError_dSnp1, dSnp1_dBn, speed);
		}
		
		//Bias update for the last neuron layer
		static void Compute_final_bias_update(Matrix<T> &last_bias_update, const Matrix<T> &error_vector, const Matrix<T> &last_dSnp1_dBn, T speed) {
			Matrix<T>::Row_Vectors_Per_Element_Multiply_AndThen_Scale(last_bias_update, error_vector, last_dSnp1_dBn, speed);
		}

	private:
		/*
		Back propagation variables:
		*/
			std::unique_ptr<Matrix<T>> error_vector, expected_vector;

			//dSnp1_dNn[n](0,i) := d(signal_nodes[n+1](0,i))/d(network_nodes[n](0,i))
			std::vector<std::unique_ptr<Matrix<T>>> dSnp1_dNn;

			//dSnp2_dSnp1[n](i,j) := d(signal_nodes[n+2](0,i))/d(signal_nodes[n+1](0,j))
			std::vector<std::unique_ptr<Matrix<T>>> dSnp2_dSnp1;

			//dSOutput_dSnp1[n](i,j) := d(signal_nodes[last](0,i))/d(signal_nodes[n+1](0,j))
			std::vector<std::unique_ptr<Matrix<T>>> dSOutput_dSnp1;

			//dError_dSnp1[n](i,j) := d(error(0,i))/d(signal_nodes[n+1](0,j))
			std::vector<std::unique_ptr<Matrix<T>>> dError_dSnp1;

			//dTotalError_dSnp1[n](0,j) := d(TotalError)/d(signal_nodes[n+1](0,j)), thios is equal to sum of every row in dError_dSnp1.
			std::vector<std::unique_ptr<Matrix<T>>> dTotalError_dSnp1;

			//dSnp1_dWn[n](j,i) := d(signal_nodes[n+1](0,i))/d(weights[n](j,i))
			std::vector<std::unique_ptr<Matrix<T>>> dSnp1_dWn;

			//dSnp1_dBn[n](i,j) := d(signal_nodes[n+1](0,i))/d(biases[n](0,j))
			std::vector<std::unique_ptr<Matrix<T>>> dSnp1_dBn;

			//Resultant updates
			std::vector<std::unique_ptr<Matrix<T>>> weight_updates;
			std::vector<std::unique_ptr<Matrix<T>>> bias_updates;
		
		/*
		Forward propagation variables:
		*/
			//Every signal_nodes[i], i!=0, represents a layer of neuron output values. signal_nodes[0] := input. All signal_nodes are row vectors
			std::vector<std::unique_ptr<Matrix<T>>> signal_nodes;

			//weights[m](i,j) := Weight from signal_nodes[m](0,i) to network_nodes[m](0,j). 
			std::vector<std::unique_ptr<Matrix<T>>> weights;

			//network_nodes[i] := signal_nodes[i] * weights[i]. All network_nodes are row vectors
			std::vector<std::unique_ptr<Matrix<T>>> network_nodes;

			//biases[i] := bias values on the neurons
			std::vector<std::unique_ptr<Matrix<T>>> biases;

			std::vector<Neuron_Type> neuron_types;

			std::vector<T> speeds;
	
	public:

		const std::vector<std::unique_ptr<Matrix<T>>> &getWeights() const{
			return weights;
		};

		const std::vector<std::unique_ptr<Matrix<T>>> &getBiases() const{
			return biases;
		};

		const std::vector<T> &getSpeeds() const {
			return speeds;
		};

		const std::vector<Neuron_Type> &getNeuronTypes() const {
			return neuron_types;
		};

		const std::vector<std::unique_ptr<Matrix<T>>> &get_signal_nodes() const {
			return signal_nodes;
		};
	};
}