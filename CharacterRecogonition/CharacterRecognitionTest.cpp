#include <array>
#include <vector>

#include "gmock\gmock.h"
#include "gtest\gtest.h"

#include "ANN.h"
#include "MinstData.h"

using namespace std;
using namespace CPPANN;

TEST(CharacterRecognition, one_hidden_layer_with_15_neurons)
{
	//Setup ANN
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_layer(0, 576)
		.set_layer(1, 20)
		.set_layer(2, 10)
		.build();



	//read raw training material
	MINSTData<double> mINSTData;
	mINSTData.read_data("./MINSTDataset/train-images.idx3-ubyte"s, "./MINSTDataset/train-labels.idx1-ubyte"s);

	


	//prepare data into a form usable by the ANN
	std::vector<std::vector<double>> training_input;
	std::vector<double> training_output;
	for (size_t idx = 0; idx < mINSTData.get_number_of_images(); ++idx) {
		training_input.at(idx) = mINSTData.get_image(idx);
	}
	





}