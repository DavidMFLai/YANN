// CPPANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <cmath>
#include <utility>
#include "ANN.h"
#include <vector>
#include <tuple>

using namespace std;
using namespace CPPANN;

int main()
{
	ANN<double> ann;
	ann.add_layer(3); //3 neurons
	ann.add_weights({
		{ 1., 2. }, //weights from the 0th neuron to the next layer
		{ 1., 2. },
		{ 1., 2. }  //weights towards the 2nd neuron to the next layer
	});
	ann.add_layer(2); //2 neurons
	ann.add_weights({
		{ 1., 2. }, //weights from the 0th neuron to the next layer
		{ 1., 2. },
	});
	ann.add_layer(2); //2 neurons

	auto result = ann.forward_propagate({ 1., 2., 3.});
	ann.back_propagate();


	return 0;
}

