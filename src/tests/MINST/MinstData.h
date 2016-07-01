#pragma once
#include <cassert>
#include <string>
#include <fstream>
#include <array>
#include <vector>

using std::string;
using std::vector;
using std::array;
using std::ifstream;
using std::ios_base;
using std::runtime_error;

using uchar = unsigned char;

static auto reverseInt = [](uint32_t i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
	return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
};

template<typename T>
class MINSTData {
public:
	array<uint32_t, 2> get_image_dimensions() const {
		return image_dimensions;
	};
	size_t get_number_of_images() const {
		return images.size();
	};

	vector<T> &get_image(size_t idx) {
		return images.at(idx);
	}

	const vector<T> &get_image(size_t idx) const {
		return images.at(idx);
	};

	uchar get_label(size_t idx) const {
		return labels.at(idx);
	};

	void read_data(const string &image_path, const string &label_path) {
		image_dimensions = Read_mnist_images(images, image_path);
		Read_mnist_labels(labels, label_path);
		assert(labels.size() == images.size());
	};

private:
	static array<uint32_t, 2> Read_mnist_images(vector<vector<T>> &output, const string &full_path) {
		ifstream file(full_path, ios_base::binary);

		if (file.is_open()) {
			//check magic number
			uint32_t magic_number = 0;
			file.read((char *)&magic_number, sizeof(uint32_t));
			magic_number = reverseInt(magic_number);
			if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

			//read dimensions and number of images;
			array<uint32_t, 2> dimensions{ 0,0 };
			uint32_t number_of_images;
			file.read((char *)&number_of_images, sizeof(uint32_t));
			number_of_images = reverseInt(number_of_images);
			file.read((char *)&dimensions[0], sizeof(uint32_t));
			dimensions[0] = reverseInt(dimensions[0]);
			file.read((char *)&dimensions[1], sizeof(uint32_t));
			dimensions[1] = reverseInt(dimensions[1]);
			output.resize(number_of_images);

			//read data
			size_t image_size;
			image_size = dimensions[0] * dimensions[1];
			for (size_t i = 0; i < number_of_images; i++) {
				output.at(i).reserve(image_size);
				for (size_t j = 0; j < image_size; ++j) {
					unsigned char raw_byte;
					file.read((char *)&raw_byte, 1);
					T data = Convert_to_normalized_T(raw_byte);
					output.at(i).push_back(data);
				}
			}
			return dimensions;
		}
		else {
			throw runtime_error("Cannot open file `" + full_path + "`!");
		}
	};

	static void Read_mnist_labels(vector<uchar> &output, const string &full_path) {
		ifstream file(full_path, ios_base::binary);

		if (file.is_open()) {
			//check magic number
			uint32_t magic_number = 0;
			file.read((char *)&magic_number, sizeof(uint32_t));
			magic_number = reverseInt(magic_number);
			if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

			//read number of labels
			uint32_t number_of_labels;
			file.read((char *)&number_of_labels, sizeof(uint32_t));
			number_of_labels = reverseInt(number_of_labels);

			//read the labels
			output.resize(number_of_labels);
			auto raw_ptr = output.data();
			file.read((char *)raw_ptr, number_of_labels);
		}
		else {
			throw runtime_error("Cannot open file `" + full_path + "`!");
		}
	};

	template<typename U = T>
	static U Convert_to_normalized_T(uchar single_data) {
		U retval = static_cast<U>(single_data);
		retval /= std::numeric_limits<uchar>::max() + 1; //the +1 allows easier division?
		return retval; 
	};
#if 0
	template<>
	static uchar Convert_to_normalized_T<uchar>(uchar single_data) {
		return single_data;
	};
#endif
private:
	array<uint32_t, 2> image_dimensions;
	vector<vector<T>> images;
	vector<uchar> labels;
};








