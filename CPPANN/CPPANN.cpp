// CPPANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <cmath>
#include <utility>
#include "ANN.h"
#include <vector>

using namespace std;
using namespace CPPANN;

int main()
{
	ANN<double> ann;
	ann.addLayer(3); //3 neurons
	ann.addWeights({
		{1., 1., 1.}, //weights towards the 0th neuron at the next layer
		{2., 2., 2.}  //weights towards the 1st neuron at the next layer
	});
	ann.addLayer(2); //2 neurons
	auto result = ann.forwardPropagate({1., 2., 3.});

	//Matrix<double> n;

	Matrix<double> m{ 
		{ 1.0, 1.1, 1.2 },
		{ 2.0, 2.1, 2.2 }
	};

	Matrix<double> n{
		{ 11.0, 11.1},
		{ 12.0, 12.1},
		{ 13.0, 13.1}
	};

	auto mn = m*n;
	auto nm = n*m;

	auto mt = m*Matrix<double>{
		{ 11.0, 11.1 },
		{ 12.0, 12.1 },
		{ 13.0, 13.1 }
	};

	return 0;
}

