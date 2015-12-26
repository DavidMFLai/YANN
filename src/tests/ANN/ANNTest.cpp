#include "ANN.h"
#include <vector>
#include <cmath>
#include <utility>
#include <vector>
#include <tuple>
#include <random>

#include "gmock\gmock.h"
#include "gtest\gtest.h"

using namespace std;
using namespace CPPANN;


TEST(Basics, mattmazur)
{
	//See http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/. But the guy didnt update the biases

	//Setup ANN
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_input_layer(2)
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 2)
		.set_output_layer(Neuron_Type::Sigmoid, 0.5, 2)
		.set_weights(0, {
			{ 0.15, 0.25 },
			{ 0.20, 0.30 },
		})
		.set_weights(1, {
			{ 0.40, 0.50 }, //weights from the 0th neuron of the present layer
			{ 0.45, 0.55 },
		})
		.set_bias(0, { 0.35, 0.35 })
		.set_bias(1, { 0.60, 0.60 })
		.build();

	//Execute ANN
	ann.forward_propagate({ 0.05, 0.10 });
	ann.back_propagate({ 0.01, 0.99});

	//Verify
	double tolerence = 0.00000001;
	auto &weights = ann.getWeights();
	EXPECT_NEAR(0.14978072, weights[0]->at(0, 0), tolerence);
	EXPECT_NEAR(0.24975114, weights[0]->at(0, 1), tolerence);
	EXPECT_NEAR(0.19956143, weights[0]->at(1, 0), tolerence);
	EXPECT_NEAR(0.29950229, weights[0]->at(1, 1), tolerence);
	EXPECT_NEAR(0.35891648, weights[1]->at(0, 0), tolerence);
	EXPECT_NEAR(0.51130127, weights[1]->at(0, 1), tolerence);
	EXPECT_NEAR(0.40866619, weights[1]->at(1, 0), tolerence);
	EXPECT_NEAR(0.56137012, weights[1]->at(1, 1), tolerence);
}

TEST(Basics, CounterCheckInPython)
{
	//Setup ANN
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_input_layer(5)
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 4)
		.set_hidden_layer(1, Neuron_Type::Sigmoid, 0.5, 3)
		.set_output_layer(Neuron_Type::Sigmoid, 0.5, 2)

		.set_weights(0, {
			{ 0.01, 0.02, 0.03, 0.04 },
			{ 0.05, 0.06, 0.07, 0.08 },
			{ 0.09, 0.10, 0.11, 0.12 },
			{ 0.13, 0.14, 0.15, 0.16 },
			{ 0.17, 0.18, 0.19, 0.20 },
		})
		.set_weights(1, {
			{ 0.25, 0.26, 0.27 },
			{ 0.28, 0.29, 0.30 },
			{ 0.31, 0.32, 0.33 },
			{ 0.34, 0.35, 0.36 },
		})
		.set_weights(2, {
			{ 0.40, 0.41 },
			{ 0.42, 0.43 },
			{ 0.44, 0.45 },
		})
		.set_bias(0, { 0.21, 0.22, 0.23, 0.24 })
		.set_bias(1, { 0.37, 0.38, 0.39 })
		.set_bias(2, { 0.46, 0.47 })
		.build();

	ann.forward_propagate({ 0.18, 0.29, 0.40, 0.51, 0.62 });
	ann.back_propagate({ 0.01, 0.99 });

	//Verify
	double tolerence = 0.00000001;
	auto &weights = ann.getWeights();
	EXPECT_NEAR(0.00987499, weights[0]->at(0, 0), tolerence);
	EXPECT_NEAR(0.0198615, weights[0]->at(0, 1), tolerence);
	EXPECT_NEAR(0.02984825, weights[0]->at(0, 2), tolerence);
	EXPECT_NEAR(0.03983527, weights[0]->at(0, 3), tolerence);
	EXPECT_NEAR(0.0497986, weights[0]->at(1, 0), tolerence);
	EXPECT_NEAR(0.05977686, weights[0]->at(1, 1), tolerence);
	EXPECT_NEAR(0.06975552, weights[0]->at(1, 2), tolerence);
	EXPECT_NEAR(0.07973461, weights[0]->at(1, 3), tolerence);
	EXPECT_NEAR(0.0897222, weights[0]->at(2, 0), tolerence);
	EXPECT_NEAR(0.09969222, weights[0]->at(2, 1), tolerence);
	EXPECT_NEAR(0.10966279, weights[0]->at(2, 2), tolerence);
	EXPECT_NEAR(0.11963394, weights[0]->at(2, 3), tolerence);
	EXPECT_NEAR(0.12964581, weights[0]->at(3, 0), tolerence);
	EXPECT_NEAR(0.13960758, weights[0]->at(3, 1), tolerence);
	EXPECT_NEAR(0.14957005, weights[0]->at(3, 2), tolerence);
	EXPECT_NEAR(0.15953327, weights[0]->at(3, 3), tolerence);
	EXPECT_NEAR(0.16956941, weights[0]->at(4, 0), tolerence);
	EXPECT_NEAR(0.17952294, weights[0]->at(4, 1), tolerence);
	EXPECT_NEAR(0.18947732, weights[0]->at(4, 2), tolerence);
	EXPECT_NEAR(0.1994326, weights[0]->at(4, 3), tolerence);
	EXPECT_NEAR(0.24780606, weights[1]->at(0, 0), tolerence);
	EXPECT_NEAR(0.25773575, weights[1]->at(0, 1), tolerence);
	EXPECT_NEAR(0.26766959, weights[1]->at(0, 2), tolerence);
	EXPECT_NEAR(0.27778027, weights[1]->at(1, 0), tolerence);
	EXPECT_NEAR(0.28770913, weights[1]->at(1, 1), tolerence);
	EXPECT_NEAR(0.29764219, weights[1]->at(1, 2), tolerence);
	EXPECT_NEAR(0.30775465, weights[1]->at(2, 0), tolerence);
	EXPECT_NEAR(0.31768269, weights[1]->at(2, 1), tolerence);
	EXPECT_NEAR(0.32761498, weights[1]->at(2, 2), tolerence);
	EXPECT_NEAR(0.33772922, weights[1]->at(3, 0), tolerence);
	EXPECT_NEAR(0.34765645, weights[1]->at(3, 1), tolerence);
	EXPECT_NEAR(0.35758797, weights[1]->at(3, 2), tolerence);
	EXPECT_NEAR(0.35310712, weights[2]->at(0, 0), tolerence);
	EXPECT_NEAR(0.4204483, weights[2]->at(0, 1), tolerence);
	EXPECT_NEAR(0.3727042, weights[2]->at(1, 0), tolerence);
	EXPECT_NEAR(0.44053808, weights[2]->at(1, 1), tolerence);
	EXPECT_NEAR(0.39230839, weights[2]->at(2, 0), tolerence);
	EXPECT_NEAR(0.46062627, weights[2]->at(2, 1), tolerence);

	auto &biases = ann.getBiases();
	EXPECT_NEAR(0.2093055, biases[0]->at(0, 0), tolerence);
	EXPECT_NEAR(0.21923054, biases[0]->at(0, 1), tolerence);
	EXPECT_NEAR(0.22915696, biases[0]->at(0, 2), tolerence);
	EXPECT_NEAR(0.23908484, biases[0]->at(0, 3), tolerence);
	EXPECT_NEAR(0.36638459, biases[1]->at(0, 0), tolerence);
	EXPECT_NEAR(0.37626872, biases[1]->at(0, 1), tolerence);
	EXPECT_NEAR(0.38615969, biases[1]->at(0, 2), tolerence);
	EXPECT_NEAR(0.39749299, biases[2]->at(0, 0), tolerence);
	EXPECT_NEAR(0.48392732, biases[2]->at(0, 1), tolerence);
}


TEST(Basics, CounterCheckInPythonTanhAndSigmoid)
{
	//Setup ANN
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_input_layer(5)
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 4)
		.set_hidden_layer(1, Neuron_Type::Tanh, 0.5, 3)
		.set_output_layer(Neuron_Type::Sigmoid, 0.5, 2)

		.set_weights(0, {
			{ 0.01, 0.02, 0.03, 0.04 },
			{ 0.05, 0.06, 0.07, 0.08 },
			{ 0.09, 0.10, 0.11, 0.12 },
			{ 0.13, 0.14, 0.15, 0.16 },
			{ 0.17, 0.18, 0.19, 0.20 },
		})
		.set_weights(1, {
			{ 0.25, 0.26, 0.27 },
			{ 0.28, 0.29, 0.30 },
			{ 0.31, 0.32, 0.33 },
			{ 0.34, 0.35, 0.36 },
		})
		.set_weights(2, {
			{ 0.40, 0.41 },
			{ 0.42, 0.43 },
			{ 0.44, 0.45 },
		})
		.set_bias(0, { 0.21, 0.22, 0.23, 0.24 })
		.set_bias(1, { 0.37, 0.38, 0.39 })
		.set_bias(2, { 0.46, 0.47 })
		.build();

	auto &output = ann.forward_propagate({ 0.18, 0.29, 0.40, 0.51, 0.62 });
	ann.back_propagate({ 0.01, 0.99 });

	//Verify
	double tolerence = 0.00000001;
	EXPECT_NEAR(0.81517039, output.at(0), tolerence);
	EXPECT_NEAR(0.82029267, output.at(1), tolerence);

	auto &weights = ann.getWeights();
	EXPECT_NEAR(0.00977151, weights[0]->at(0, 0), tolerence);
	EXPECT_NEAR(0.01974683, weights[0]->at(0, 1), tolerence);
	EXPECT_NEAR(0.02972259, weights[0]->at(0, 2), tolerence);
	EXPECT_NEAR(0.03969884, weights[0]->at(0, 3), tolerence);
	EXPECT_NEAR(0.04963188, weights[0]->at(1, 0), tolerence);
	EXPECT_NEAR(0.05959211, weights[0]->at(1, 1), tolerence);
	EXPECT_NEAR(0.06955307, weights[0]->at(1, 2), tolerence);
	EXPECT_NEAR(0.0795148, weights[0]->at(1, 3), tolerence);
	EXPECT_NEAR(0.08949225, weights[0]->at(2, 0), tolerence);
	EXPECT_NEAR(0.09943739, weights[0]->at(2, 1), tolerence);
	EXPECT_NEAR(0.10938354, weights[0]->at(2, 2), tolerence);
	EXPECT_NEAR(0.11933076, weights[0]->at(2, 3), tolerence);
	EXPECT_NEAR(0.12935262, weights[0]->at(3, 0), tolerence);
	EXPECT_NEAR(0.13928268, weights[0]->at(3, 1), tolerence);
	EXPECT_NEAR(0.14921402, weights[0]->at(3, 2), tolerence);
	EXPECT_NEAR(0.15914672, weights[0]->at(3, 3), tolerence);
	EXPECT_NEAR(0.16921299, weights[0]->at(4, 0), tolerence);
	EXPECT_NEAR(0.17912796, weights[0]->at(4, 1), tolerence);
	EXPECT_NEAR(0.18904449, weights[0]->at(4, 2), tolerence);
	EXPECT_NEAR(0.19896268, weights[0]->at(4, 3), tolerence);
	EXPECT_NEAR(0.24582719, weights[1]->at(0, 0), tolerence);
	EXPECT_NEAR(0.25585576, weights[1]->at(0, 1), tolerence);
	EXPECT_NEAR(0.26589693, weights[1]->at(0, 2), tolerence);
	EXPECT_NEAR(0.27577813, weights[1]->at(1, 0), tolerence);
	EXPECT_NEAR(0.28580703, weights[1]->at(1, 1), tolerence);
	EXPECT_NEAR(0.2958487, weights[1]->at(1, 2), tolerence);
	EXPECT_NEAR(0.30572941, weights[1]->at(2, 0), tolerence);
	EXPECT_NEAR(0.31575864, weights[1]->at(2, 1), tolerence);
	EXPECT_NEAR(0.32580079, weights[1]->at(2, 2), tolerence);
	EXPECT_NEAR(0.33568104, weights[1]->at(3, 0), tolerence);
	EXPECT_NEAR(0.3457106, weights[1]->at(3, 1), tolerence);
	EXPECT_NEAR(0.35575322, weights[1]->at(3, 2), tolerence);
	EXPECT_NEAR(0.35145125, weights[2]->at(0, 0), tolerence);
	EXPECT_NEAR(0.42001165, weights[2]->at(0, 1), tolerence);
	EXPECT_NEAR(0.37071558, weights[2]->at(1, 0), tolerence);
	EXPECT_NEAR(0.44016336, weights[2]->at(1, 1), tolerence);
	EXPECT_NEAR(0.39002023, weights[2]->at(2, 0), tolerence);
	EXPECT_NEAR(0.46030675, weights[2]->at(2, 1), tolerence);

	auto &biases = ann.getBiases();
	EXPECT_NEAR(0.20873063, biases[0]->at(0, 0), tolerence);
	EXPECT_NEAR(0.21859348, biases[0]->at(0, 1), tolerence);
	EXPECT_NEAR(0.22845886, biases[0]->at(0, 2), tolerence);
	EXPECT_NEAR(0.2383269, biases[0]->at(0, 3), tolerence);
	EXPECT_NEAR(0.36312358, biases[1]->at(0, 0), tolerence);
	EXPECT_NEAR(0.37317065, biases[1]->at(0, 1), tolerence);
	EXPECT_NEAR(0.38323851, biases[1]->at(0, 2), tolerence);
	EXPECT_NEAR(0.39934344, biases[2]->at(0, 0), tolerence);
	EXPECT_NEAR(0.4825085, biases[2]->at(0, 1), tolerence);
}


TEST(Basics, CounterCheckInPythonTanhAndSigmoidWithDifferentSpeeds)
{
	//Setup ANN
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_input_layer(5)
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 4)
		.set_hidden_layer(1, Neuron_Type::Tanh, 0.4, 3)
		.set_output_layer(Neuron_Type::Sigmoid, 0.3, 2)

		.set_weights(0, {
			{ 0.01, 0.02, 0.03, 0.04 },
			{ 0.05, 0.06, 0.07, 0.08 },
			{ 0.09, 0.10, 0.11, 0.12 },
			{ 0.13, 0.14, 0.15, 0.16 },
			{ 0.17, 0.18, 0.19, 0.20 },
		})
		.set_weights(1, {
			{ 0.25, 0.26, 0.27 },
			{ 0.28, 0.29, 0.30 },
			{ 0.31, 0.32, 0.33 },
			{ 0.34, 0.35, 0.36 },
		})
		.set_weights(2, {
			{ 0.40, 0.41 },
			{ 0.42, 0.43 },
			{ 0.44, 0.45 },
		})
		.set_bias(0, { 0.21, 0.22, 0.23, 0.24 })
		.set_bias(1, { 0.37, 0.38, 0.39 })
		.set_bias(2, { 0.46, 0.47 })
		.build();

	auto &output = ann.forward_propagate({ 0.18, 0.29, 0.40, 0.51, 0.62 });
	ann.back_propagate({ 0.01, 0.99 });

	//Verify
	double tolerence = 0.00000001;
	EXPECT_NEAR(0.81517039, output.at(0), tolerence);
	EXPECT_NEAR(0.82029267, output.at(1), tolerence);

	auto &weights = ann.getWeights();
	EXPECT_NEAR(0.00977151, weights[0]->at(0, 0), tolerence);
	EXPECT_NEAR(0.01974683, weights[0]->at(0, 1), tolerence);
	EXPECT_NEAR(0.02972259, weights[0]->at(0, 2), tolerence);
	EXPECT_NEAR(0.03969884, weights[0]->at(0, 3), tolerence);
	EXPECT_NEAR(0.04963188, weights[0]->at(1, 0), tolerence);
	EXPECT_NEAR(0.05959211, weights[0]->at(1, 1), tolerence);
	EXPECT_NEAR(0.06955307, weights[0]->at(1, 2), tolerence);
	EXPECT_NEAR(0.0795148, weights[0]->at(1, 3), tolerence);
	EXPECT_NEAR(0.08949225, weights[0]->at(2, 0), tolerence);
	EXPECT_NEAR(0.09943739, weights[0]->at(2, 1), tolerence);
	EXPECT_NEAR(0.10938354, weights[0]->at(2, 2), tolerence);
	EXPECT_NEAR(0.11933076, weights[0]->at(2, 3), tolerence);
	EXPECT_NEAR(0.12935262, weights[0]->at(3, 0), tolerence);
	EXPECT_NEAR(0.13928268, weights[0]->at(3, 1), tolerence);
	EXPECT_NEAR(0.14921402, weights[0]->at(3, 2), tolerence);
	EXPECT_NEAR(0.15914672, weights[0]->at(3, 3), tolerence);
	EXPECT_NEAR(0.16921299, weights[0]->at(4, 0), tolerence);
	EXPECT_NEAR(0.17912796, weights[0]->at(4, 1), tolerence);
	EXPECT_NEAR(0.18904449, weights[0]->at(4, 2), tolerence);
	EXPECT_NEAR(0.19896268, weights[0]->at(4, 3), tolerence);
	EXPECT_NEAR(0.24666175, weights[1]->at(0, 0), tolerence);
	EXPECT_NEAR(0.25668461, weights[1]->at(0, 1), tolerence);
	EXPECT_NEAR(0.26671755, weights[1]->at(0, 2), tolerence);
	EXPECT_NEAR(0.27662251, weights[1]->at(1, 0), tolerence);
	EXPECT_NEAR(0.28664563, weights[1]->at(1, 1), tolerence);
	EXPECT_NEAR(0.29667896, weights[1]->at(1, 2), tolerence);
	EXPECT_NEAR(0.30658353, weights[1]->at(2, 0), tolerence);
	EXPECT_NEAR(0.31660692, weights[1]->at(2, 1), tolerence);
	EXPECT_NEAR(0.32664063, weights[1]->at(2, 2), tolerence);
	EXPECT_NEAR(0.33654483, weights[1]->at(3, 0), tolerence);
	EXPECT_NEAR(0.34656848, weights[1]->at(3, 1), tolerence);
	EXPECT_NEAR(0.35660258, weights[1]->at(3, 2), tolerence);
	EXPECT_NEAR(0.37087075, weights[2]->at(0, 0), tolerence);
	EXPECT_NEAR(0.41600699, weights[2]->at(0, 1), tolerence);
	EXPECT_NEAR(0.39042935, weights[2]->at(1, 0), tolerence);
	EXPECT_NEAR(0.43609801, weights[2]->at(1, 1), tolerence);
	EXPECT_NEAR(0.41001214, weights[2]->at(2, 0), tolerence);
	EXPECT_NEAR(0.45618405, weights[2]->at(2, 1), tolerence);

	auto &biases = ann.getBiases();
	EXPECT_NEAR(0.20873063, biases[0]->at(0, 0), tolerence);
	EXPECT_NEAR(0.21859348, biases[0]->at(0, 1), tolerence);
	EXPECT_NEAR(0.22845886, biases[0]->at(0, 2), tolerence);
	EXPECT_NEAR(0.2383269, biases[0]->at(0, 3), tolerence);
	EXPECT_NEAR(0.36449886, biases[1]->at(0, 0), tolerence);
	EXPECT_NEAR(0.37453652, biases[1]->at(0, 1), tolerence);
	EXPECT_NEAR(0.38459081, biases[1]->at(0, 2), tolerence);
	EXPECT_NEAR(0.42360607, biases[2]->at(0, 0), tolerence);
	EXPECT_NEAR(0.4775051, biases[2]->at(0, 1), tolerence);
}

TEST(Basics, Accending_and_decending)
{
	//Setup ANN
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_input_layer(5)
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 4)
		.set_hidden_layer(1, Neuron_Type::Sigmoid, 0.5, 3)
		.set_output_layer(Neuron_Type::Sigmoid, 0.5, 2)
		.set_weights(0, {
			{ 0.01, 0.02, 0.03, 0.04 },
			{ 0.05, 0.06, 0.07, 0.08 },
			{ 0.09, 0.10, 0.11, 0.12 },
			{ 0.13, 0.14, 0.15, 0.16 },
			{ 0.17, 0.18, 0.19, 0.20 },
		})
		.set_weights(1, {
			{ 0.25, 0.26, 0.27 },
			{ 0.28, 0.29, 0.30 },
			{ 0.31, 0.32, 0.33 },
			{ 0.34, 0.35, 0.36 },
		})
		.set_weights(2, {
			{ 0.40, 0.41 },
			{ 0.42, 0.43 },
			{ 0.44, 0.45 },
		})
		.set_bias(0, { 0.21, 0.22, 0.23, 0.24 })
		.set_bias(1, { 0.37, 0.38, 0.39 })
		.set_bias(2, { 0.46, 0.47 })
		.build();

	std::vector<double> low_to_high_input = { 0.18, 0.29, 0.40, 0.51, 0.62 };
	std::vector<double> low_to_high_expected_result = { 0.01, 0.99 };
	std::vector<double> high_to_low_input = { 0.62, 0.51, 0.40, 0.29, 0.18 };
	std::vector<double> high_to_low_expected_result = { 0.99, 0.01 };

	for (int i = 0; i < 1000000; i++) {
		ann.forward_propagate(low_to_high_input);
		ann.back_propagate(low_to_high_expected_result);
		ann.forward_propagate(high_to_low_input);
		ann.back_propagate(high_to_low_expected_result);
	}

	std::vector<double> low_to_high_output = ann.forward_propagate(low_to_high_input);
	std::vector<double> high_to_low_output = ann.forward_propagate(high_to_low_input);

	double tolerence = 0.05;
	EXPECT_NEAR(0.01, low_to_high_output[0], tolerence);
	EXPECT_NEAR(0.99, low_to_high_output[1], tolerence);
	EXPECT_NEAR(0.99, high_to_low_output[0], tolerence);
	EXPECT_NEAR(0.01, high_to_low_output[1], tolerence);
}

TEST(Basics, XOR_RANDOM_SIGMOID)
{
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder
		.set_input_layer(2)
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 2)
		.set_output_layer(Neuron_Type::Sigmoid, 0.5, 1)
		.build();

	std::vector<double> true_true_input{ 1., 1. };
	std::vector<double> true_true_expected_result{ 0. };

	std::vector<double> false_false_input{ 0., 0. };
	std::vector<double> false_false_expected_result{ 0. };

	std::vector<double> false_true_input{ 0., 1. };
	std::vector<double> false_true_expected_result{ 1. };

	std::vector<double> true_false_input{ 1., 0. };
	std::vector<double> true_false_expected_result{ 1. };

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


TEST(ANNBuilder_Basics, ANNBuilder_Basics)
{
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_input_layer(5)
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 4)
		.set_hidden_layer(1, Neuron_Type::Sigmoid, 0.4, 3)
		.set_output_layer(Neuron_Type::Tanh, 0.3, 2)
		.build();

	auto &biases = ann.getBiases();
	auto &neuron_counts = ann.get_signal_nodes();
	auto &weight_matrices = ann.getWeights();
	auto &neuron_types = ann.getNeuronTypes();
	auto &speeds = ann.getSpeeds();

	EXPECT_EQ(3, biases.size());
	EXPECT_EQ(4, biases.at(0)->getRowLength());
	EXPECT_EQ(3, biases.at(1)->getRowLength());
	EXPECT_EQ(2, biases.at(2)->getRowLength());

	EXPECT_EQ(4, neuron_counts.size());

	EXPECT_EQ(3, weight_matrices.size());
	EXPECT_EQ(5, weight_matrices.at(0)->getColumnLength());
	EXPECT_EQ(4, weight_matrices.at(0)->getRowLength());
	EXPECT_EQ(4, weight_matrices.at(1)->getColumnLength());
	EXPECT_EQ(3, weight_matrices.at(1)->getRowLength());
	EXPECT_EQ(3, weight_matrices.at(2)->getColumnLength());
	EXPECT_EQ(2, weight_matrices.at(2)->getRowLength());

	EXPECT_EQ(3, neuron_types.size());
	EXPECT_EQ(Neuron_Type::Sigmoid, neuron_types.at(0));
	EXPECT_EQ(Neuron_Type::Sigmoid, neuron_types.at(1));
	EXPECT_EQ(Neuron_Type::Tanh, neuron_types.at(2));

	EXPECT_EQ(3, speeds.size());
	double tolerance = 0.00000001;
	EXPECT_NEAR(0.5, speeds.at(0), 0.00000001);
	EXPECT_NEAR(0.4, speeds.at(1), 0.00000001);
	EXPECT_NEAR(0.3, speeds.at(2), 0.00000001);
}