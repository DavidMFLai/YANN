#pragma once
#include "Matrix.h"

namespace CPPANN {
	template<typename T>
	class Signal_Nodes : public Matrix<T> {
	public:
		Signal_Nodes(size_t rowCount) 
			: Matrix<T>(rowCount, 1)
		{};

		void set_values(std::vector<T> &&values) {
			assert(values.size() == Matrix<T>::getElems().size());
			assert(values.size() == Matrix<T>::getDimensions()[0]);
			Matrix<T>::getElems() = std::move(values);
		}

		std::vector<T> &get_values() {
			return Matrix<T>::getElems();
		}
	};

	template<typename T>
	class Network_Nodes : public Matrix<T> {
	public:
		Network_Nodes(size_t rowCount)
			: Matrix<T>(rowCount, 1)
		{};

		Network_Nodes &operator=(const Matrix<T> & mat) {
			assert(this->getDimensions() == mat.getDimensions());
			Matrix<T>::operator=(mat);
			return *this;
		}

		Signal_Nodes<T> apply_sigmoid() {
			Signal_Nodes<T> retval{ Matrix<T>::getDimensions()[0]};

			auto &elems = Matrix<T>::getElems();
			for (size_t i = 0; i < elems.size(); i++) {
				retval(i,0) = sigmoid(Matrix<T>::operator()(i,0));
			}

			return retval;
		};

		std::vector<T> &get_values() {
			return Matrix<T>::getElems();
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
	class ANN {
	public:
		ANN() = default;
		void add_layer(uint64_t size) {
			signal_nodes.push_back(Signal_Nodes<T>{size});
			if (signal_nodes.size() > 1) {
				network_nodes.push_back(Network_Nodes<T>{size});
			}
		}

		void add_weights(Weight_Matrix<T> matrix) {
			weights.push_back(std::move(matrix));
		}

		const std::vector<T> &forward_propagate(std::vector<T> &&input) {
			signal_nodes[0].set_values(std::move(input));

			for (int i = 0; i < weights.size(); i++) {
				network_nodes[i] = weights[i] * signal_nodes[i];
				signal_nodes[i + 1] = network_nodes[i].apply_sigmoid();
			}

			return signal_nodes.back().get_values();
		}

	private:
		std::vector<Signal_Nodes<T>> signal_nodes;
		std::vector<Weight_Matrix<T>> weights;
		std::vector<Network_Nodes<T>> network_nodes;
	};

}