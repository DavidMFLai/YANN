#include "MinstReader.h"

using namespace std;

//code is adapted from: http://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
Images read_mnist_images(string full_path, size_t &number_of_images, size_t &image_size) {
	auto reverseInt = [](uint32_t i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((size_t)c1 << 24) + ((size_t)c2 << 16) + ((size_t)c3 << 8) + c4;
	};

	ifstream file(full_path, std::ios_base::binary);

	if (file.is_open()) {
		size_t magic_number = 0;
		size_t n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(int32_t));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(int32_t));
		number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(int32_t));
		n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(int32_t));
		n_cols = reverseInt(n_cols);

		Images images{ n_rows, n_cols, number_of_images };
		image_size = n_rows * n_cols;
		for (size_t i = 0; i < number_of_images; i++) {
			images.get_image(i).resize(image_size);
			auto raw_ptr = images.get_image(i).data();
			file.read((char *)raw_ptr, image_size);
		}
		return images;
	}
	else {
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}

uchar* read_mnist_labels(string full_path, int& number_of_labels) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	ifstream file(full_path);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		uchar* _dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	}
	else {
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}