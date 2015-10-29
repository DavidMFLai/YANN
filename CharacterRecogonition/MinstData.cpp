#include "MinstData2.h"
//code is adapted from: http://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

using namespace std;

static auto reverseInt = [](uint32_t i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
	return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
};

//Static function
array<uint32_t, 2> MINSTData2::Read_mnist_images(vector<vector<uchar>> &output, const string &full_path) {
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
			output.at(i).resize(image_size);
			auto raw_ptr = output.at(i).data();
			file.read((char *)raw_ptr, image_size);
		}
		return dimensions;
	}
	else {
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}

//Static function
void MINSTData2::Read_mnist_labels(vector<uchar> &output, const string &full_path) {
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
}

size_t MINSTData2::get_number_of_images() const {
	return images.size();
};

array<uint32_t, 2> MINSTData2::get_image_dimensions() const {
	return image_dimensions;
}

vector<uchar> &MINSTData2::get_image(size_t idx) {
	return images.at(idx);
}

uchar MINSTData2::get_label(size_t idx) const{
	return labels.at(idx);
}

const vector<uchar> &MINSTData2::get_image(size_t idx) const{
	return images.at(idx);
}

void MINSTData2::read_data(const string &image_path, const string &label_path) {
	image_dimensions = Read_mnist_images(images, image_path);
	Read_mnist_labels(labels, label_path);
	assert(labels.size() == images.size());
}