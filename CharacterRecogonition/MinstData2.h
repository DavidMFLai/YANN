#pragma once
#include <cassert>
#include <string>
#include <fstream>
#include <array>
#include <vector>

using std::string;
using std::vector;
using std::array;
typedef unsigned char uchar;

class MINSTData2 {
public:
	size_t get_number_of_images() const;
	array<uint32_t, 2> get_image_dimensions() const;
	const vector<uchar> &get_image(size_t idx) const;
	uchar get_label(size_t idx) const;
	vector<uchar> &get_image(size_t idx);
	void read_data(const string &image_path, const string &label_path);

private:
	static array<uint32_t, 2> Read_mnist_images(vector<vector<uchar>> &output, const string &full_path);
	static void Read_mnist_labels(vector<uchar> &output, const string &full_path);
	array<uint32_t, 2> image_dimensions;
	vector<vector<uchar>> images;
	vector<uchar> labels;
};