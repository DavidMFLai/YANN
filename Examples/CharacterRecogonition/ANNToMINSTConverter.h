#pragma once
#include "ANN.h"
#include "MinstData.h"

namespace Converter {
	void Convert_label_to_ANN_output_data(Matrix<double> & ann_output_data, uchar mINST_label);
	uchar Convert_ANN_output_data_to_label(const std::vector<double>& ann_output_data);
}