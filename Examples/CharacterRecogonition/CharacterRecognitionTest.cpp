#include <array>
#include <vector>
#include <iostream>

#include "gmock\gmock.h"
#include "gtest\gtest.h"

#include "ANN.h"
#include "MinstData.h"

using namespace std;
using namespace CPPANN;

static void Convert_label_to_ANN_output_data(Matrix<double> & ann_output_data, uchar mINST_label) {
	ann_output_data.zero();
	switch (mINST_label) {
	case 0:
		ann_output_data(0, 0) = 1.;
		break;
	case 1:
		ann_output_data(0, 1) = 1.;
		break;
	case 2:
		ann_output_data(0, 2) = 1.;
		break;
	case 3:
		ann_output_data(0, 3) = 1.;
		break;
	case 4:
		ann_output_data(0, 4) = 1.;
		break;
	case 5:
		ann_output_data(0, 5) = 1.;
		break;
	case 6:
		ann_output_data(0, 6) = 1.;
		break;
	case 7:
		ann_output_data(0, 7) = 1.;
		break;
	case 8:
		ann_output_data(0, 8) = 1.;
		break;
	case 9:
		ann_output_data(0, 9) = 1.;
		break;
	}
}

static uchar Convert_ANN_output_data_to_label(const std::vector<double>& ann_output_data) {
	uchar retval;
	double max = std::numeric_limits<double>::lowest();
	for (size_t idx = 0; idx < ann_output_data.size(); ++idx) {
		if (ann_output_data.at(idx) > max) {
			retval = static_cast<uchar>(idx);
			max = ann_output_data.at(idx);
		}
	}
	return retval;
}

TEST(CharacterRecognition, one_hidden_layer_with_15_neurons)
{
	string train_images_full_path = "../common/MINST/MINSTDataset/train-images.idx3-ubyte";
	string train_labels_full_path = "../common/MINST/MINSTDataset/train-labels.idx1-ubyte";
	string test_images_full_path = "../common/MINST/MINSTDataset/t10k-images.idx3-ubyte";
	string test_labels_full_path = "../common/MINST/MINSTDataset/t10k-labels.idx1-ubyte";


	//read raw training material
	MINSTData<double> mINSTData;
	mINSTData.read_data(train_images_full_path, train_labels_full_path);

	//Setup ANN
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_input_layer(mINSTData.get_number_of_images())
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 15)
		.set_output_layer(Neuron_Type::Sigmoid, 0.5, 10)
		.build();

	//Train with first 5000 only
	Matrix<double> training_output_data{ 1, 10 };
	for (size_t j = 0; j < 10; j++) {
		for (size_t idx = 0; idx < 5000; idx++) {
			auto &training_input_data = mINSTData.get_image(idx);
			auto training_output_data_raw = mINSTData.get_label(idx);
			Convert_label_to_ANN_output_data(training_output_data, training_output_data_raw);
			ann.forward_propagate(training_input_data);
			ann.back_propagate(training_output_data);
		}
		std::cout << "";
	}

	//read raw testing material
	MINSTData<double> mINSTData_test;
	mINSTData_test.read_data(test_images_full_path, test_labels_full_path);

	//Test
	size_t correct_count = 0;
	size_t total_count = 0;
	Matrix<double> testing_output_data{ 1, 10 };
	for (size_t idx = 0; idx < mINSTData_test.get_number_of_images(); idx++) {
		auto &test_input_data = mINSTData_test.get_image(idx);
		std::vector<double> ann_result = ann.forward_propagate(test_input_data);
		uchar result = Convert_ANN_output_data_to_label(ann_result);

		uchar label = mINSTData_test.get_label(idx);
		
		if (result == label) {
			correct_count++;
		}
		total_count++;
		
		std::cout << "Correct Ratio = " << correct_count << '/' << total_count << std::endl;

	}

}