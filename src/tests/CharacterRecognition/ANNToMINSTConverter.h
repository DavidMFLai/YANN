#pragma once
#include "ANN.h"
#include "MinstData.h"

namespace Converter {
	template<typename T>
	void Convert_label_to_ANN_output_data(std::vector<T> &ann_output_data, uchar mINST_label) {
		std::fill(ann_output_data.begin(), ann_output_data.end(), static_cast<T>(0));
		switch (mINST_label) {
		case 0:
			ann_output_data.at(0) = static_cast<T>(1.);
			break;
		case 1:
			ann_output_data.at(1) = static_cast<T>(1.);
			break;
		case 2:
			ann_output_data.at(2) = static_cast<T>(1.);
			break;
		case 3:
			ann_output_data.at(3) = static_cast<T>(1.);
			break;
		case 4:
			ann_output_data.at(4) = static_cast<T>(1.);
			break;
		case 5:
			ann_output_data.at(5) = static_cast<T>(1.);
			break;
		case 6:
			ann_output_data.at(6) = static_cast<T>(1.);
			break;
		case 7:
			ann_output_data.at(7) = static_cast<T>(1.);
			break;
		case 8:
			ann_output_data.at(8) = static_cast<T>(1.);
			break;
		case 9:
			ann_output_data.at(9) = static_cast<T>(1.);
			break;
		}
	}

	template<typename T>
	uchar Convert_ANN_output_data_to_label(const std::vector<T> &ann_output_data) {
		uchar retval;
		double max = std::numeric_limits<T>::lowest();
		for (size_t idx = 0; idx < ann_output_data.size(); ++idx) {
			if (ann_output_data.at(idx) > max) {
				retval = static_cast<uchar>(idx);
				max = ann_output_data.at(idx);
			}
		}
		return retval;
	}
}