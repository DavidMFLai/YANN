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


	return 0;
}

