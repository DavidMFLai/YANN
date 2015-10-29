#include <fstream>

#include "gmock\gmock.h"
#include "gtest\gtest.h"

#include "ANN.h"
#include "MinstReader.h"

using namespace std;
using namespace CPPANN;




TEST(MINST, ReadFiles)
{
	size_t number_of_images, image_size;

	Images images = read_mnist_images("C:/Users/lai_m_000/Documents/Visual Studio 2015/Projects/CPPANN/CharacterRecogonition/MINSTDataset/train-images.idx3-ubyte"s, number_of_images, image_size);


}


int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}