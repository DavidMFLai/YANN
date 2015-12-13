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

namespace {
	MATCHER_P(FloatNearPointwise, tol, "Out of range") {
		return (std::get<0>(arg) > std::get<1>(arg) - tol && std::get<0>(arg) < std::get<1>(arg) + tol);
	}

	void set_sum_of_rows_test_internal(const std::vector<std::vector<float>> &input_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.create(input_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.create(1, input_data.at(0).size())};
		Matrix<float>::Sum_of_rows(*output_ref, *input_ref);
		auto output_ref_as_vector = output_ref->getElems();

		OpenCLMatrixBuilder<float> openCLMatrixBuilder(input_data.size() * input_data.at(0).size());
		auto input = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.create(input_data) };
		auto output = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.create(1, input_data.at(0).size()) };
		Matrix<float>::Sum_of_rows(*output, *input);
		auto output_as_vector = output->getElems();

		float tolerance = 0.000001;
		EXPECT_EQ(output_ref_as_vector.size(), output_as_vector.size());
		for (size_t i = 0; i < output_as_vector.size(); i++) {
			EXPECT_FLOAT_EQ(output_ref_as_vector.at(i), output_as_vector.at(i), tolerance);
		}
	}

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

	TEST(BasicOperations, set_to_sum_of_rows_power_of_2) {
		std::vector<std::vector<float>> input_data{
			{ 1.1f, 1.2f },
			{ 21.2f, 25.3f },
			{ 31.3f, 35.4f },
			{ 41.4f, 45.7f },
		};
		set_sum_of_rows_test_internal(input_data);
	}

	TEST(BasicOperations, set_to_sum_of_rows_not_power_of_2) {
		std::vector<std::vector<float>> input_data{
			{ 1.1f, 1.2f },
			{ 21.2f, 25.3f },
			{ 31.3f, 35.4f },
			{ 41.4f, 45.7f },
			{ 31.3f, 35.4f },
			{ 41.4f, 45.7f }
		};
		set_sum_of_rows_test_internal(input_data);
	}

	TEST(BasicOperations, set_to_sum_of_rows_very_long) {
		std::vector<std::vector<float>> input_data;

		for (int idx = 0; idx < 10000; idx++) {
			input_data.push_back({ 1.f, 2.f });
		}

		set_sum_of_rows_test_internal(input_data);
	}
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
