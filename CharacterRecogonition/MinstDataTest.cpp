#include <array>

#include "gmock\gmock.h"
#include "gtest\gtest.h"

#include "ANN.h"
#include "MinstData2.h"

using namespace std;

TEST(MINST, BasicChecks)
{
	MINSTData2 mINSTData;
	mINSTData.read_data("./MINSTDataset/train-images.idx3-ubyte"s, "./MINSTDataset/train-labels.idx1-ubyte"s);
	
	//verify
	EXPECT_EQ(60000, mINSTData.get_number_of_images());
	array<uint32_t, 2>  expectation{ 28, 28 };
	EXPECT_EQ(expectation, mINSTData.get_image_dimensions());
	EXPECT_EQ(expectation[0] * expectation[1], mINSTData.get_image(0).size());
}