#include "ANNToMINSTConverter.h"

void Converter::Convert_label_to_ANN_output_data(Matrix<double> &ann_output_data, uchar mINST_label) {
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