#include "ANNToMINSTConverter.h"

void Converter::Convert_label_to_ANN_output_data(std::vector<double> &ann_output_data, uchar mINST_label) {
	std::fill(ann_output_data.begin(), ann_output_data.end(), 0);
	switch (mINST_label) {
	case 0:
		ann_output_data.at(0) = 1.;
		break;
	case 1:
		ann_output_data.at(1) = 1.;
		break;
	case 2:
		ann_output_data.at(2) = 1.;
		break;
	case 3:
		ann_output_data.at(3) = 1.;
		break;
	case 4:
		ann_output_data.at(4) = 1.;
		break;
	case 5:
		ann_output_data.at(5) = 1.;
		break;
	case 6:
		ann_output_data.at(6) = 1.;
		break;
	case 7:
		ann_output_data.at(7) = 1.;
		break;
	case 8:
		ann_output_data.at(8) = 1.;
		break;
	case 9:
		ann_output_data.at(9) = 1.;
		break;
	}
}

uchar Converter::Convert_ANN_output_data_to_label(const std::vector<double>& ann_output_data)
{
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