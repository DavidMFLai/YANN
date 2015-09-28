#pragma once
#include "Matrix.h"

namespace CPPANN {
	template<typename T>
	class NetworkNodes;

	template<typename T>
	class SignalNodes : public Matrix<T>
	{
	public:
		SignalNodes(size_t rowCount) 
			: Matrix<T>(rowCount, 1)
		{};

		SignalNodes &operator=(const Matrix<T> &mat) {
			assert(this->getDimensions() == mat.getDimensions());
			Matrix<T>::operator=(mat);
			return *this;
		}

		std::vector<T> &get_values() {
			return Matrix<T>::getElems();
		}
	};

	template<typename T>
	class NetworkNodes : public Matrix<T>
	{
	public:
		NetworkNodes(size_t rowCount)
			: Matrix<T>(rowCount, 1)
		{};

		NetworkNodes &operator=(const Matrix<T> & mat) {
			assert(this->getDimensions() == mat.getDimensions());
			Matrix<T>::operator=(mat);
			return *this;
		}

		SignalNodes<T> apply_sigmoid() {
			SignalNodes<T> retval{ Matrix<T>::getDimensions()[0]};

			auto &elems = Matrix<T>::getElems();
			for (size_t i = 0; i < elems.size(); i++) {
				retval(i,0) = sigmoid(this->operator()(i,0));
			}

			return retval;
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
			signalNodes.push_back(SignalNodes<T>{size});
			if (signalNodes.size() > 1) {
				networkNodes.push_back(NetworkNodes<T>{size});
			}
		}

		void add_weights(Weight_Matrix<T> matrix) {
			weights.push_back(std::move(matrix));
		}

		const std::vector<T> &forward_propagate(const std::vector<T> &input) {
			signalNodes[0] = input;

			for (int i = 0; i < weights.size(); i++) {
				networkNodes[i] = weights[i] * signalNodes[i];
				signalNodes[i + 1] = networkNodes[i].apply_sigmoid();
			}

			return signalNodes.back().get_values();
		}

	private:
		std::vector<SignalNodes<T>> signalNodes;
		std::vector<Weight_Matrix<T>> weights;
		std::vector<NetworkNodes<T>> networkNodes;
	};

}