// ANNTest.cpp : Defines the entry point for the console application.
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
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_layer(0, 2)
		.set_bias(0, { 0.35, 0.35 })
		.set_layer(1, 2)
		.set_bias(1, { 0.60, 0.60 })
		.set_weights(0, {
			{ 0.15, 0.25 },
			{ 0.20, 0.30 },
		})
		.set_layer(2, 2)
		.set_weights(1, {
			{ 0.40, 0.50 }, //weights from the 0th neuron of the present layer
			{ 0.45, 0.55 },
		})
		.build();

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

TEST(Basics, XOR_SIGMOID)
{
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_layer(0, 2)
		.set_layer(1, 2)
		.set_weights(0, {
			{ 0.5, -0.7 },
			{ -0.8, 0.6 }
		})
		.set_bias(0, { 0.01, -0.9 })
		.set_weights(1, {
			{ 2. }, //weights from the 0th neuron of the present layer
			{ 3. },
		})
		.set_layer(2, 1)
		.set_bias(1, { -0.8 })
		.build();

	std::vector<double> true_true_input{ 1., 1. };
	Matrix<double> true_true_expected_result{ {0.} };

	std::vector<double> false_false_input{ 0., 0. };
	Matrix<double> false_false_expected_result{ { 0. } };
	
	std::vector<double> false_true_input{ 0., 1. };
	Matrix<double> false_true_expected_result{ { 1. } };

	std::vector<double> true_false_input{ 1., 0. };
	Matrix<double> true_false_expected_result{ { 1. } };

	std::vector<double> true_true_result, false_true_result, true_false_result, false_false_result;
	for (int i = 0; i < 1000000; i++) {
		true_true_result = ann.forward_propagate(true_true_input);
		ann.back_propagate(true_true_expected_result);
		false_false_result = ann.forward_propagate(false_false_input);
		ann.back_propagate(false_false_expected_result);
		false_true_result = ann.forward_propagate(false_true_input);
		ann.back_propagate(false_true_expected_result);
		true_false_result = ann.forward_propagate(true_false_input);
		ann.back_propagate(true_false_expected_result);
	}
	double tolerence = 0.05;
	EXPECT_NEAR(0., true_true_result[0], tolerence);
	EXPECT_NEAR(0., false_false_result[0], tolerence);
	EXPECT_NEAR(1., false_true_result[0], tolerence);
	EXPECT_NEAR(1., true_false_result[0], tolerence);
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}