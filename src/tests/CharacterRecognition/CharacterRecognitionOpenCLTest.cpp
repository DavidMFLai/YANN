#include <array>
#include <vector>
#include <iostream>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ANN.h"
#include "MinstData.h"
#include "ANNToMINSTConverter.h"
#include "OpenCLMatrix.h"
#include "OpenCLMatrixBuilder.h"

using namespace std;
using namespace YANN;
using namespace Converter;

TEST(CharacterRecognitionOpenCL, one_hidden_layer_with_15_neurons)
{
	string train_images_full_path = MINSTDATA_ROOT + "train-images.idx3-ubyte"s;
	string train_labels_full_path = MINSTDATA_ROOT + "train-labels.idx1-ubyte"s;
	string test_images_full_path = MINSTDATA_ROOT + "t10k-images.idx3-ubyte"s;
	string test_labels_full_path = MINSTDATA_ROOT + "t10k-labels.idx1-ubyte"s;


	//read raw training material
	MINSTData<float> mINSTData;
	mINSTData.read_data(train_images_full_path, train_labels_full_path);

	//setup OpenCLMatrixBuilder
	auto openCLMatrixBuilder = std::make_unique<OpenCLMatrixBuilder<float>>();
	openCLMatrixBuilder->set_max_matrix_element_count(mINSTData.get_image_dimensions().at(0) *  mINSTData.get_image_dimensions().at(1) * 15);
	//openCLMatrixBuilder->set_platform_name_contains("Intel");
	//openCLMatrixBuilder->set_device_type(CL_DEVICE_TYPE_CPU);

	//Setup ANN
	ANNBuilder<float> ann_builder;
	auto ann = ann_builder.set_input_layer(mINSTData.get_image_dimensions().at(0) *  mINSTData.get_image_dimensions().at(1))
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 15)
		.set_output_layer(Neuron_Type::Sigmoid, 0.5, 10)
		.set_matrix_builder(std::move(openCLMatrixBuilder))
		.build();

	//Train with first 5000 only
	std::vector<float> training_output_data(10);
	for (size_t j = 0; j < 10; j++) {
		for (size_t idx = 0; idx < 5000; idx++) {
			auto &training_input_data = mINSTData.get_image(idx);
			auto training_output_data_raw = mINSTData.get_label(idx);
			Convert_label_to_ANN_output_data(training_output_data, training_output_data_raw);
			ann.forward_propagate_no_output(training_input_data);
			ann.back_propagate(training_output_data);
		}
		std::cout << "";
	}

	//read raw testing material
	MINSTData<float> mINSTData_test;
	mINSTData_test.read_data(test_images_full_path, test_labels_full_path);

	//Test
	size_t correct_count = 0;
	size_t total_count = 0;
	std::vector<float> testing_output_data(10);
	for (size_t idx = 0; idx < mINSTData_test.get_number_of_images(); idx++) {
		auto &test_input_data = mINSTData_test.get_image(idx);
		std::vector<float> ann_result = ann.forward_propagate(test_input_data);
		uchar result = Convert_ANN_output_data_to_label(ann_result);

		uchar label = mINSTData_test.get_label(idx);
		
		if (result == label) {
			correct_count++;
		}
		total_count++;
		
		std::cout << "Correct Ratio = " << correct_count << '/' << total_count << '\n';

	}
	
}
