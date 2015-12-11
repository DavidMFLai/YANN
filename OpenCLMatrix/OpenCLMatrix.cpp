// OpenCLMatrix.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "gmock\gmock.h"
#include "gtest/gtest.h"

#include "Matrix.h"
#include "include\OpenCLMatrix.h"
#include "include\OpenCLMatrixBuilder.h"

TEST(BasicOperations, create_from_dimensions) {
	
	OpenCLMatrixBuilder<float> builder;
	auto m = builder.create(3, 4);

	const auto dimensions = m->getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 3, 4 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, create_from_initializer_lists) {

	OpenCLMatrixBuilder<float> builder;
	auto m = builder.create({ 
		{ 1.1f, 1.2f, 1.3f, 1.4f },
		{ 2.1f, 2.2f, 2.3f, 2.4f },
		{ 3.1f, 3.2f, 3.3f, 3.4f },
	});

	const auto dimensions = m->getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 3, 4 };
	EXPECT_EQ(dimensions, expectedDimensions);

	auto elems = m->getElems();
	std::vector<float> expected_elems({
		{ 1.1f, 1.2f, 1.3f, 1.4f, 2.1f, 2.2f, 2.3f, 2.4f, 3.1f, 3.2f, 3.3f, 3.4f },
	});

	EXPECT_EQ(elems, expected_elems);
}

TEST(BasicOperations, create_from_vectors) {

	OpenCLMatrixBuilder<float> builder;
	auto m = builder.create(std::vector<std::vector<float>>{
		{ 1.1f, 1.2f, 1.3f, 1.4f },
		{ 2.1f, 2.2f, 2.3f, 2.4f },
		{ 3.1f, 3.2f, 3.3f, 3.4f },
	});

	const auto dimensions = m->getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 3, 4 };
	EXPECT_EQ(dimensions, expectedDimensions);

	auto elems = m->getElems();
	std::vector<float> expected_elems({
		{ 1.1f, 1.2f, 1.3f, 1.4f, 2.1f, 2.2f, 2.3f, 2.4f, 3.1f, 3.2f, 3.3f, 3.4f },
	});

	EXPECT_EQ(elems, expected_elems);
}


TEST(BasicOperations, create_row_vector_matrix) {

	OpenCLMatrixBuilder<float> builder;
	auto m = builder.createRowMatrix(std::vector<float>{
		{ 1.1f, 1.2f, 1.3f, 1.4f },
	});

	const auto dimensions = m->getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 1, 4 };
	EXPECT_EQ(dimensions, expectedDimensions);

	auto elems = m->getElems();
	std::vector<float> expected_elems({
		{ 1.1f, 1.2f, 1.3f, 1.4f },
	});

	EXPECT_EQ(elems, expected_elems);
}


TEST(BasicOperations, set_to_sum_of_rows) {

	OpenCLMatrixBuilder<float> builder;
	auto input = std::unique_ptr<Matrix<float>>{ builder.create({
		{ 1.1f, 1.2f },
		{ 21.2f, 25.3f },
		{ 31.3f, 35.4f },
		{ 41.4f, 45.7f },
	}) };

	auto output = std::unique_ptr<Matrix<float>>{ builder.create(1, 2) };
	auto output_before_running_as_vector = output->getElems();
	std::cout << "output before running: ";
	for (auto data : output_before_running_as_vector) {
		std::cout << data << " ";
	}
	std::cout << std::endl;

	Matrix<float>::Sum_of_rows(*output, *input);

	auto output_as_vector = output->getElems();
	auto input_as_vector = input->getElems();

	std::cout << "input: ";
	for (auto data : input_as_vector) {
		std::cout << data << " ";
	}
	std::cout << std::endl;

	std::cout << "output: ";
	for (auto data : output_as_vector) {
		std::cout << data << " ";
	}

}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
