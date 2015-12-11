// OpenCLMatrix.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "gmock\gmock.h"
#include "gtest/gtest.h"

#include "Matrix.h"
#include "include\OpenCLMatrix.h"
#include "include\OpenCLMatrixBuilder.h"
#include "..\Matrix\include\ReferenceMatrix.h"
#include "..\Matrix\include\ReferenceMatrixBuilder.h"

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
	std::vector<std::vector<float>> input_data{
		{ 1.1f, 1.2f },
		{ 21.2f, 25.3f },
		{ 31.3f, 35.4f },
		{ 41.4f, 45.7f },
	};

	ReferenceMatrixBuilder<float> referenceMatrixBuilder;
	auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.create(input_data) };
	auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.create(1, 2) };
	Matrix<float>::Sum_of_rows(*output_ref, *input_ref);
	auto output_ref_as_vector = output_ref->getElems();

	OpenCLMatrixBuilder<float> openCLMatrixBuilder;
	auto input = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.create(input_data) };
	auto output = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.create(1, 2) };
	Matrix<float>::Sum_of_rows(*output, *input);
	auto output_as_vector = output->getElems();

	EXPECT_EQ(output_as_vector, output_ref_as_vector);
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
