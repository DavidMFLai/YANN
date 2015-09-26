// CPPANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <cmath>
#include <utility>
#include "Eigen/Dense"
#include "Matrix.h"
#include <vector>

//using uint64 = uint64_t;
//using Eigen::MatrixXd;
//using Eigen::VectorXd;
//using hiddenLayer_t = std::tuple<uint64, std::vector<uint64>>;
//
//using layer_t = std::pair<VectorXd, MatrixXd>;

using namespace std;

namespace CPPANN {

	
	//class ANN 
	//{
	//public:
	//	ANN();
	//	void addLayer(uint64 count);
	//	void forwardPropagate();

	//private:
	//	VectorXd lastHalfLayer;
	//	vector<pair<VectorXd, MatrixXd>> layers; //last layer will have no weight matrix
	//};

	//ANN::ANN() 
	//{
	//	
	//}

	//void ANN::addLayer(uint64 count)
	//{
	//	if (layers.size() != 0) 
	//	{
	//		layers.push_back(make_pair(lastHalfLayer, MatrixXd{ count, lastHalfLayer.size() }));
	//	}
	//	lastHalfLayer = VectorXd{ count };
	//}

	//void ANN::forwardPropagate()
	//{

	//}

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

		const  std::vector<T> &forwardPropagate(const std::vector<T> &input) {
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

using namespace CPPANN;


#include "Matrix.h"
int main()
{
	//ANN ann;
	//ann.addLayer(3);
	//ann.addLayer(4);

	ANN<double> ann;
	ann.addLayer(3); //3 neurons
	ann.addWeights(std::vector<std::vector<double>>{
		{1., 1., 1.},
		{2., 2., 2.}
	});
	ann.addLayer(2); //2 neurons
	auto result = ann.forwardPropagate(std::vector<double>{1., 2., 3.});
	return 0;
}

