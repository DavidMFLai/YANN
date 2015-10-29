#pragma once
#include <string>
#include <fstream>
#include <array>
#include <vector>

using std::string;
using std::vector;
using std::array;
typedef unsigned char uchar;

class Images {
public:
	Images(size_t n_rows, size_t n_columns, size_t number_of_images)
		:image_dimensions{ n_rows , n_columns },
		data(number_of_images)
	{};

	size_t get_number_of_images() const {
		return data.size();
	};

	auto get_image_dimensions() const {
		return image_dimensions;
	}

	vector<uchar> &get_image(size_t idx) {
		return data.at(idx);
	}

private:
	array<size_t, 2> image_dimensions;
	vector<vector<uchar>> data;
};

Images read_mnist_images(string full_path, size_t& number_of_images, size_t& image_size);
uchar* read_mnist_labels(string full_path, int& number_of_labels);