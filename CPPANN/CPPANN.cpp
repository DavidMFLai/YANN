// CPPANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <cmath>
#include <utility>
#include "ANN.h"
#include <vector>
#include <tuple>
#include <random>

#include "gmock\gmock.h"
#include "gtest/gtest.h"

using namespace std;
using namespace CPPANN;

TEST(Basics, mattmazur)
{
	//See http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

	//Setup ANN
	ANN<double> ann;
	ann.add_layer(2); //2 neurons
	ann.add_weights({ 
		{ 0.15, 0.25 }, 
		{ 0.20, 0.30 },
	});
	ann.add_layer(2); //2 neurons
	ann.add_bias({ 0.35, 0.35 });
	ann.add_weights({
		{ 0.40, 0.50 }, //weights from the 0th neuron of the present layer
		{ 0.45, 0.55 },
	});
	ann.add_layer(2); //1 neuron
	ann.add_bias({ 0.60, 0.60 });

	//Execute ANN
	ann.forward_propagate({ 0.05, 0.10 });
	ann.back_propagate(Matrix<double>{ { 0.01, 0.99} });

	//Verify
	double tolerence = 0.00000001;
	auto &weights = ann.getWeights();
	EXPECT_NEAR(0.14978072, weights[0](0, 0), tolerence);
	EXPECT_NEAR(0.24975114, weights[0](0, 1), tolerence);
	EXPECT_NEAR(0.19956143, weights[0](1, 0), tolerence);
	EXPECT_NEAR(0.29950229, weights[0](1, 1), tolerence);
	EXPECT_NEAR(0.35891648, weights[1](0, 0), tolerence);
	EXPECT_NEAR(0.51130127, weights[1](0, 1), tolerence);
	EXPECT_NEAR(0.40866619, weights[1](1, 0), tolerence);
	EXPECT_NEAR(0.56137012, weights[1](1, 1), tolerence);
}

TEST(Basics, DISABLED_RANDXOR)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1, 1);

	ANN<double> ann;
	ann.add_layer(2); //2 neurons
	ann.add_weights({
		{ dis(gen), dis(gen) },
		{ dis(gen), dis(gen) },
	});

	ann.add_layer(2); //2 neurons
	ann.add_bias({ dis(gen), dis(gen) });

	ann.add_weights({
		{ dis(gen) }, //weights from the 0th neuron of the present layer
		{ dis(gen) },
	});

	ann.add_layer(1); //1 neuron
	ann.add_bias({ dis(gen) });

	std::vector<double> tmp1, tmp2, tmp3, tmp4;
	for (int i = 0; i < 100000; i++) {
		tmp1 = ann.forward_propagate({ 0.99, 0.98 });
		ann.back_propagate(Matrix<double>{ {0.} });
		tmp4 = ann.forward_propagate({ 0.01, 0.01 });
		ann.back_propagate(Matrix<double>{ {0.} });
		tmp2 = ann.forward_propagate({ 0.01, 0.99 });
		ann.back_propagate(Matrix<double>{ {1.} });
		tmp3 = ann.forward_propagate({ 0.99, 0.01 });
		ann.back_propagate(Matrix<double>{ {1.} });
	}

	tmp1 = ann.forward_propagate({ 0.99, 0.99 });
	ann.back_propagate(Matrix<double>{ {0.} });

	tmp2 = ann.forward_propagate({ 0.01, 0.99 });
	ann.back_propagate(Matrix<double>{ {1.} });

	tmp3 = ann.forward_propagate({ 0.99, 0.01 });
	ann.back_propagate(Matrix<double>{ {1.} });

	tmp4 = ann.forward_propagate({ 0.01, 0.01 });
	ann.back_propagate(Matrix<double>{ {0.} });

	auto &biases = ann.getBiases();
	auto &weights = ann.getWeights();
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}