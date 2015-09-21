// CPPANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <cmath>
#include <tuple>
#include "Eigen/Dense"

using uint64 = unsigned long long;
using Eigen::MatrixXd;
using hiddenLayer_t = std::tuple<uint64, std::vector<uint64>>;

namespace CPPANN {
	double sigmoid(double x) {
		return 1 / (1 + exp(-x));
	};
	
	class ANN {
	public:
		ANN();
		void addLayer(uint64 count);
		


	private:
		std::vector<hiddenLayer_t> hiddenLayers;
		std::vector<MatrixXd> weightMatrices;
	};

	ANN::ANN() {
		
	};

	void ANN::addLayer(uint64 count) {
		auto newLayer = std::make_tuple(count, std::vector<uint64>(count));
		if (hiddenLayers.size() > 0) {
			auto previousLayer = hiddenLayers.back();
			MatrixXd weightMatrix(std::get<0>(newLayer), std::get<0>(previousLayer));
			weightMatrices.push_back(weightMatrix);
		}
		hiddenLayers.push_back(newLayer);
	}


}

using namespace CPPANN;

int main()
{
	ANN ann;
	ann.addLayer(3);
	ann.addLayer(4);

	return 0;
}

