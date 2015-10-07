// CPPANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <cmath>
#include <utility>
#include "ANN.h"
#include <vector>
#include <tuple>

#include "gmock\gmock.h"
#include "gtest/gtest.h"

using namespace std;
using namespace CPPANN;

TEST(Basics, DISABLED_temp)
{
	ANN<double> ann;
	ann.add_layer(5); //5 neurons
	ann.add_weights({
		{ 1.11, 1.12, 1.13, 1.14 }, 
		{ 1.21, 1.22, 1.23, 1.24 },
		{ 1.31, 1.32, 1.33, 1.34 },
		{ 1.41, 1.42, 1.43, 1.44 },
		{ 1.51, 1.52, 1.53, 1.54 }
	});
	ann.add_layer(4); //4 neurons
	ann.add_weights({
		{ 2.11, 2.12, 2.13 },
		{ 2.21, 2.22, 2.23 },
		{ 2.31, 2.32, 2.33 },
		{ 2.41, 2.42, 2.43 },
	});
	ann.add_layer(3); //3 neurons
	ann.add_weights({
		{ 3.11, 3.12 },
		{ 3.21, 3.22 },
		{ 3.31, 3.32 },
	});
	ann.add_layer(2); //2 neurons
	 
	for (int i = 0; i < 1; i++) {
		ann.forward_propagate({ 0.11, 0.12, 0.13, 0.14, 0.15 });
		ann.back_propagate(Matrix<double>{ {1., 0.} });
	}
}

TEST(Basics, XOR)
{
	ANN<double> ann;
	ann.add_layer(2); //2 neurons
	ann.add_weights({
		{ 1.11, 1.12 }, 
		{ 1.21, 1.22 },
	});

	ann.add_layer(2); //2 neurons
	ann.add_bias({ 5.1, 5.2 });
	
	ann.add_weights({
		{ 2.11 }, //weights from the 0th neuron of the present layer
		{ 2.21 },
	});

	ann.add_layer(1); //1 neuron
	ann.add_bias({ 8.1});

	for (int i = 0; i < 1; i++) {
		ann.forward_propagate({ 0.99, 0.01 });
		ann.back_propagate(Matrix<double>{ {1.} });
	}

}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}