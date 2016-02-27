// OpenCLMatrix.cpp : Defines the entry point for the console application.
//

#include <functional>

#include "gmock\gmock.h"
#include "gtest/gtest.h"

#include "Matrix.h"
#include "OpenCLMatrix.h"
#include "OpenCLMatrixBuilder.h"
#include "ReferenceMatrix.h"
#include "ReferenceMatrixBuilder.h"

namespace YANN{
	void set_sum_of_rows_test_internal(const std::vector<std::vector<float>> &input_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(input_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(1, input_data.at(0).size()) };
		Matrix<float>::Sum_of_rows(*output_ref, *input_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(input_data.size() * input_data.at(0).size());
		auto input_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(input_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(1, input_data.at(0).size()) };
		Matrix<float>::Sum_of_rows(*output_cl, *input_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void set_to_sum_of_test_internal(const std::vector<std::vector<float>> &lhs_data, const std::vector<std::vector<float>> &rhs_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto lhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(lhs_data) };
		auto rhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(rhs_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(lhs_data.size(), lhs_data.at(0).size()) };
		Matrix<float>::Add(*output_ref, *lhs_ref, *rhs_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(lhs_data.size() * lhs_data.at(0).size());
		auto lhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(lhs_data) };
		auto rhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(rhs_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(lhs_data.size(), lhs_data.at(0).size()) };
		Matrix<float>::Add(*output_cl, *lhs_cl, *rhs_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void per_Row_Multiply_test_internal(const std::vector<std::vector<float>> &multipliers_data, const std::vector<std::vector<float>> &multiplicand_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto multipliers_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multipliers_data) };
		auto multiplicand_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multiplicand_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multiplicand_ref->getColumnLength(), multiplicand_ref->getRowLength()) };
		Matrix<float>::Per_Row_Multiply(*output_ref, *multipliers_ref, *multiplicand_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(multiplicand_data.size() * multiplicand_data.at(0).size());
		auto multipliers_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multipliers_data) };
		auto multiplicand_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multiplicand_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multiplicand_cl->getColumnLength(), multiplicand_cl->getRowLength()) };
		Matrix<float>::Per_Row_Multiply(*output_cl, *multipliers_cl, *multiplicand_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void per_column_multiply_and_then_scale_test_internal(const std::vector<std::vector<float>> &multipliers_data, const std::vector<std::vector<float>> &multiplicand_data, float scale) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto multipliers_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multipliers_data) };
		auto multiplicand_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multiplicand_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multiplicand_ref->getColumnLength(), multiplicand_ref->getRowLength()) };
		Matrix<float>::Per_Column_Multiply_AndThen_Scale(*output_ref, *multipliers_ref, *multiplicand_ref, scale);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(multiplicand_data.size() * multiplicand_data.at(0).size());
		auto multipliers_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multipliers_data) };
		auto multiplicand_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multiplicand_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multiplicand_cl->getColumnLength(), multiplicand_cl->getRowLength()) };
		Matrix<float>::Per_Column_Multiply_AndThen_Scale(*output_cl, *multipliers_cl, *multiplicand_cl, scale);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void per_column_multiply_and_then_transpose_test_internal(const std::vector<std::vector<float>> &multipliers_data, const std::vector<std::vector<float>> & multiplicand_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto multipliers_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multipliers_data) };
		auto multiplicand_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multiplicand_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(multiplicand_ref->getRowLength(), multiplicand_ref->getColumnLength()) };
		Matrix<float>::Per_Column_Multiply_AndThen_Transpose(*output_ref, *multipliers_ref, *multiplicand_ref);


		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(multiplicand_data.size() * multiplicand_data.at(0).size());
		auto multipliers_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multipliers_data) };
		auto multiplicand_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multiplicand_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(multiplicand_cl->getRowLength(), multiplicand_cl->getColumnLength()) };
		Matrix<float>::Per_Column_Multiply_AndThen_Transpose(*output_cl, *multipliers_cl, *multiplicand_cl);

		EXPECT_EQ(*output_ref, *output_cl);

	}

	void row_vectors_per_element_multiply_and_then_scale_test_internal(const std::vector<float> &lhs_data, const std::vector<float> &rhs_data, float scale) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto lhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.buildRowMatrix(lhs_data) };
		auto rhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.buildRowMatrix(rhs_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(rhs_ref->getColumnLength(), rhs_ref->getRowLength()) };
		Matrix<float>::Row_Vectors_Per_Element_Multiply_AndThen_Scale(*output_ref, *lhs_ref, *rhs_ref, scale);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(rhs_data.size());
		auto lhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.buildRowMatrix(lhs_data) };
		auto rhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.buildRowMatrix(rhs_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(rhs_cl->getColumnLength(), rhs_cl->getRowLength()) };
		Matrix<float>::Row_Vectors_Per_Element_Multiply_AndThen_Scale(*output_cl, *lhs_cl, *rhs_cl, scale);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void copy_test_internal(const std::vector<std::vector<float>> &input_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(input_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(input_ref->getColumnLength(), input_ref->getRowLength()) };
		Matrix<float>::Copy(*output_ref, *input_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(input_data.size() * input_data.at(0).size());
		auto input_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(input_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(input_cl->getColumnLength(), input_cl->getRowLength()) };
		Matrix<float>::Copy(*output_cl, *input_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void outer_product_test_internal(const std::vector<std::vector<float>> &lhs_data, const std::vector<std::vector<float>> &rhs_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto lhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(lhs_data) };
		auto rhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(rhs_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(lhs_ref->getRowLength(), rhs_ref->getRowLength()) };
		Matrix<float>::Outer_product(*output_ref, *lhs_ref, *rhs_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(lhs_data.at(0).size() * rhs_data.at(0).size());
		auto lhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(lhs_data) };
		auto rhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(rhs_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(lhs_cl->getRowLength(), rhs_cl->getRowLength()) };
		Matrix<float>::Outer_product(*output_cl, *lhs_cl, *rhs_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void subtract_by_test_internal(const std::vector<std::vector<float>> &output_data, const std::vector<std::vector<float>> &input_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(input_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(output_data) };
		Matrix<float>::Subtract_By(*output_ref, *input_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(output_data.at(0).size() * output_data.size());
		auto input_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(input_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(output_data) };
		Matrix<float>::Subtract_By(*output_cl, *input_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void per_element_sigmoid_test_internal(const std::vector<std::vector<float>> &output_data, const std::vector<std::vector<float>> &input_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(input_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(output_data) };
		Matrix<float>::Per_Element_Sigmoid(*output_ref, *input_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(output_data.at(0).size() * output_data.size());
		auto input_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(input_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(output_data) };
		Matrix<float>::Per_Element_Sigmoid(*output_cl, *input_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void per_element_sigmoid_prime_test_internal(const std::vector<std::vector<float>> &output_data, const std::vector<std::vector<float>> &input_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(input_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(output_data) };
		Matrix<float>::Per_Element_Sigmoid_Prime(*output_ref, *input_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(output_data.at(0).size() * output_data.size());
		auto input_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(input_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(output_data) };
		Matrix<float>::Per_Element_Sigmoid_Prime(*output_cl, *input_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void per_element_tanh_test_internal(const std::vector<std::vector<float>> &output_data, const std::vector<std::vector<float>> &input_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(input_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(output_data) };
		Matrix<float>::Per_Element_Tanh(*output_ref, *input_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(output_data.at(0).size() * output_data.size());
		auto input_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(input_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(output_data) };
		Matrix<float>::Per_Element_Tanh(*output_cl, *input_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void per_element_tanh_prime_test_internal(const std::vector<std::vector<float>> &output_data, const std::vector<std::vector<float>> &input_data) {
		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto input_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(input_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(output_data) };
		Matrix<float>::Per_Element_Tanh_Prime(*output_ref, *input_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(output_data.at(0).size() * output_data.size());
		auto input_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(input_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(output_data) };
		Matrix<float>::Per_Element_Tanh_Prime(*output_cl, *input_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void set_to_difference_of_test_internal(const std::vector<std::vector<float>> &output_data, 
		const std::vector<std::vector<float>> &lhs_data,
		const std::vector<std::vector<float>> &rhs_data) {

		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto lhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(lhs_data) };
		auto rhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(rhs_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(output_data) };
		Matrix<float>::Minus(*output_ref, *lhs_ref, *rhs_ref);

		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(output_data.at(0).size() * output_data.size());
		auto lhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(lhs_data) };
		auto rhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(rhs_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(output_data) };
		Matrix<float>::Minus(*output_cl, *lhs_cl, *rhs_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	void set_to_product_of_test_internal(const std::vector<std::vector<float>> &output_data, 
		const std::vector<std::vector<float>> &lhs_data,
		const std::vector<std::vector<float>> &rhs_data) 
	{

		ReferenceMatrixBuilder<float> referenceMatrixBuilder;
		auto lhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(lhs_data) };
		auto rhs_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(rhs_data) };
		auto output_ref = std::unique_ptr<Matrix<float>>{ referenceMatrixBuilder.build(output_data) };
		Matrix<float>::Multiply(*output_ref, *lhs_ref, *rhs_ref);

		//get data counts and use that for openCLMatrixBuilder
		std::vector<size_t> counts;
		counts.push_back(output_ref->getColumnLength() * output_ref->getRowLength());
		counts.push_back(lhs_ref->getColumnLength() * lhs_ref->getRowLength());
		counts.push_back(rhs_ref->getColumnLength() * rhs_ref->getRowLength());
		std::sort(counts.begin(), counts.end(), std::greater<size_t>()); //sort in decending order
				
		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(counts.at(0) * counts.at(1));
		auto lhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(lhs_data) };
		auto rhs_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(rhs_data) };
		auto output_cl = std::unique_ptr<Matrix<float>>{ openCLMatrixBuilder.build(output_data) };
		Matrix<float>::Multiply(*output_cl, *lhs_cl, *rhs_cl);

		EXPECT_EQ(*output_ref, *output_cl);
	}

	TEST(BasicOperations, per_element_sigmoid) {
		size_t column_length = 50;
		size_t row_length = 101;

		std::vector<std::vector<float>> output_data(column_length);
		for (int idy = 0; idy < output_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				output_data.at(idy).push_back(9.3f * idx * idy);
			}
		}

		std::vector<std::vector<float>> input_data(column_length);
		for (int idy = 0; idy < input_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				input_data.at(idy).push_back(1.2f * idx * std::log(1.f + idy));
			}
		}

		per_element_sigmoid_test_internal(output_data, input_data);
	}

	TEST(BasicOperations, per_element_sigmoid_prime) {
		size_t column_length = 50;
		size_t row_length = 101;

		std::vector<std::vector<float>> output_data(column_length);
		for (int idy = 0; idy < output_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				output_data.at(idy).push_back(9.3f * idx * idy);
			}
		}

		std::vector<std::vector<float>> input_data(column_length);
		for (int idy = 0; idy < input_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				input_data.at(idy).push_back(1.2f * idx * std::log(1.f + idy));
			}
		}

		per_element_sigmoid_prime_test_internal(output_data, input_data);
	}

	TEST(BasicOperations, per_element_tanh) {
		size_t column_length = 50;
		size_t row_length = 101;

		std::vector<std::vector<float>> output_data(column_length);
		for (int idy = 0; idy < output_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				output_data.at(idy).push_back(9.3f * idx * idy);
			}
		}

		std::vector<std::vector<float>> input_data(column_length);
		for (int idy = 0; idy < input_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				input_data.at(idy).push_back(1.2f * idx * std::log(1.f + idy));
			}
		}

		per_element_tanh_test_internal(output_data, input_data);
	}

	TEST(BasicOperations, per_element_tanh_prime) {
		size_t column_length = 50;
		size_t row_length = 101;

		std::vector<std::vector<float>> output_data(column_length);
		for (int idy = 0; idy < output_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				output_data.at(idy).push_back(9.3f * idx * idy);
			}
		}

		std::vector<std::vector<float>> input_data(column_length);
		for (int idy = 0; idy < input_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				input_data.at(idy).push_back(1.2f * idx * std::log(1.f + idy));
			}
		}

		per_element_tanh_prime_test_internal(output_data, input_data);
	}


	TEST(BasicOperations, create_from_dimensions) {
		OpenCLMatrixBuilder<float> builder;
		auto m = builder.build(3, 4);

		const auto dimensions = m->getDimensions();
		const std::array<size_t, 2> expectedDimensions{ 3, 4 };
		EXPECT_EQ(dimensions, expectedDimensions);
	}

	TEST(BasicOperations, create_from_initializer_lists) {

		OpenCLMatrixBuilder<float> builder;
		auto m = builder.build({
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
		auto m = builder.build(std::vector<std::vector<float>>{
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
		auto m = builder.buildRowMatrix(std::vector<float>{
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

	TEST(BasicOperations, set_to_sum_of_very_long) {
		std::vector<std::vector<float>> lhs_data;
		std::vector<std::vector<float>> rhs_data;
		for (int idx = 0; idx < 10000; idx++) {
			lhs_data.push_back({ 1.f, 2.f });
			rhs_data.push_back({ 3.f, 5.f });
		}

		set_to_sum_of_test_internal(lhs_data, rhs_data);
	}

	TEST(BasicOperations, per_Row_Multiply_very_long) {
		std::vector<std::vector<float>> multipliers_data(1);
		for (int idx = 0; idx < 1000; idx++) {
			multipliers_data.at(0).push_back(7.f * idx);
		}

		std::vector<std::vector<float>> multiplicand_data(500);
		for (int idy = 0; idy < multiplicand_data.size(); idy++) {
			for (int idx = 0; idx < 1000; idx++) {
				multiplicand_data.at(idy).push_back(9.f * idx * idy);
			}
		}

		per_Row_Multiply_test_internal(multipliers_data, multiplicand_data);
	}

	TEST(BasicOperations, Per_Column_Multiply_AndThen_Scale_long) {
		std::vector<std::vector<float>> multipliers_data(1);
		for (int idx = 0; idx < 1001; idx++) {
			multipliers_data.at(0).push_back(7.f * idx * std::log(1.f + idx));
		}

		std::vector<std::vector<float>> multiplicand_data(500);
		for (int idy = 0; idy < multiplicand_data.size(); idy++) {
			for (int idx = 0; idx < 1001; idx++) {
				multiplicand_data.at(idy).push_back(9.f * idx * idy);
			}
		}
		per_column_multiply_and_then_scale_test_internal(multipliers_data, multiplicand_data, 3.5f);
	}

	TEST(BasicOperations, Row_Vectors_Per_Element_Multiply_AndThen_Scale_short) {
		std::vector<float> lhs;
		for (int idx = 0; idx < 2; idx++) {
			lhs.push_back(7.6f * idx * std::log(1.f + idx));
		}

		std::vector<float> rhs;
		for (int idx = 0; idx < 2; idx++) {
			rhs.push_back(8.6f * -idx * std::log(5.f + idx));
		}

		row_vectors_per_element_multiply_and_then_scale_test_internal(lhs, rhs, 4.6f);
	}

	TEST(BasicOperations, Row_Vectors_Per_Element_Multiply_AndThen_Scale_long) {
		std::vector<float> lhs;
		for (int idx = 0; idx < 10001; idx++) {
			lhs.push_back(7.6f * idx * std::log(1.f + idx));
		}

		std::vector<float> rhs;
		for (int idx = 0; idx < 10001; idx++) {
			rhs.push_back(8.6f * -idx * std::log(5.f + idx));
		}
		row_vectors_per_element_multiply_and_then_scale_test_internal(lhs, rhs, 4.6f);
	}

	TEST(BasicOperations, copy_long) {
		std::vector<std::vector<float>> input_data(500);
		for (int idy = 0; idy < input_data.size(); idy++) {
			for (int idx = 0; idx < 1001; idx++) {
				input_data.at(idy).push_back(9.f * idx * idy);
			}
		}
		copy_test_internal(input_data);
	}

	TEST(BasicOperations, copy_short) {
		std::vector<std::vector<float>> input_data(50);
		for (int idy = 0; idy < input_data.size(); idy++) {
			for (int idx = 0; idx < 11; idx++) {
				input_data.at(idy).push_back(9.f * idx * idy);
			}
		}
		copy_test_internal(input_data);
	}


	TEST(BasicOperations, outer_product_long) {
		std::vector<std::vector<float>> lhs_data(1);
		for (int idx = 0; idx < 1005; idx++) {
			lhs_data.at(0).push_back(1.5f * idx * std::log(1.f + idx));
		}

		std::vector<std::vector<float>> rhs_data(1);
		for (int idx = 0; idx < 1001; idx++) {
			rhs_data.at(0).push_back(9.f * idx);
		}

		outer_product_test_internal(lhs_data, rhs_data);
	}

	TEST(BasicOperations, outer_product_short) {
		std::vector<std::vector<float>> lhs_data(1);
		for (int idx = 0; idx < 10; idx++) {
			lhs_data.at(0).push_back(1.5f * idx * std::log(1.f + idx));
		}

		std::vector<std::vector<float>> rhs_data(1);
		for (int idx = 0; idx < 11; idx++) {
			rhs_data.at(0).push_back(9.f * idx);
		}

		outer_product_test_internal(lhs_data, rhs_data);
	}

	TEST(BasicOperations, copy_from_vector) {
		const size_t size_x = 33;
		const size_t size_y = 37;
		std::vector<float> input_data;
		for (int idx = 0; idx < size_x * size_y; idx++) {
			input_data.push_back(1.5f * idx * std::log(1.f + idx));
		}

		//create opencl_matrix
		OpenCLMatrixBuilder<float> openCLMatrixBuilder;
		openCLMatrixBuilder.set_max_matrix_element_count(input_data.size());
		auto opencl_matrix = openCLMatrixBuilder.build(size_x, size_y);

		//copy input data into opencl_matrix
		Matrix<float>::Copy_from_vector(*opencl_matrix, input_data);

		//get the data back from the opencl_matrix
		auto elems_in_opencl_matrix = opencl_matrix->getElems();

		EXPECT_EQ(input_data.size(), elems_in_opencl_matrix.size());
		for (size_t i = 0; i < elems_in_opencl_matrix.size(); i++) {
			ASSERT_FLOAT_EQ(input_data.at(i), elems_in_opencl_matrix.at(i));
		}
	}
	
	TEST(BasicOperations, subtract_by_short) {
		size_t column_length = 50;
		size_t row_length = 101;

		std::vector<std::vector<float>> output_data(column_length);
		for (int idy = 0; idy < output_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				output_data.at(idy).push_back(9.3f * idx * idy);
			}
		}

		std::vector<std::vector<float>> input_data(column_length);
		for (int idy = 0; idy < input_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				input_data.at(idy).push_back(1.2f * idx * std::log(1.f + idy));
			}
		}
		subtract_by_test_internal(output_data, input_data);
	}

	TEST(BasicOperations, subtract_by_long) {
		size_t column_length = 500;
		size_t row_length = 1001;

		std::vector<std::vector<float>> output_data(column_length);
		for (int idy = 0; idy < output_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				output_data.at(idy).push_back(9.3f * idx * idy);
			}
		}
		
		std::vector<std::vector<float>> input_data(column_length);
		for (int idy = 0; idy < input_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				input_data.at(idy).push_back(1.2f * idx * std::log(1.f + idy));
			}
		}

		subtract_by_test_internal(output_data, input_data);
	}

	TEST(BasicOperations, set_to_difference_of_long) {
		size_t column_length = 500;
		size_t row_length = 1001;

		std::vector<std::vector<float>> output_data(column_length);
		for (int idy = 0; idy < output_data.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				output_data.at(idy).push_back(9.3f * idx * idy);
			}
		}

		std::vector<std::vector<float>> lhs(column_length);
		for (int idy = 0; idy < lhs.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				lhs.at(idy).push_back(1.2f * idx * std::log(1.f + idy));
			}
		}

		std::vector<std::vector<float>> rhs(column_length);
		for (int idy = 0; idy < rhs.size(); idy++) {
			for (int idx = 0; idx < row_length; idx++) {
				rhs.at(idy).push_back(1.9f * idx * std::log(5.2f + idy));
			}
		}

		set_to_difference_of_test_internal(output_data, lhs, rhs);
	}

	TEST(BasicOperations, set_to_product_of) {
		size_t M = 50;
		size_t K = 101;
		size_t N = 201;

		std::vector<std::vector<float>> lhs_data;
		lhs_data.resize(M);
		for (int idy = 0; idy < M; idy++) {
			for (int idx = 0; idx < K; idx++) {
				lhs_data.at(idy).push_back(9.3f * idx * idy + idx + idy + 2.3f);
			}
		}

		std::vector<std::vector<float>> rhs_data;
		rhs_data.resize(K);
		for (int idy = 0; idy < K; idy++) {
			for (int idx = 0; idx < N; idx++) {
				rhs_data.at(idy).push_back(1.2f * idx + std::log(1.f + idy));
			}
		}

		std::vector<std::vector<float>> output_data;
		output_data.resize(M);
		for (int idy = 0; idy < M; idy++) {
			for (int idx = 0; idx < N; idx++) {
				output_data.at(idy).push_back(0.0f);
			}
		}

		set_to_product_of_test_internal(output_data, lhs_data, rhs_data);
	}


	TEST(BasicOperations, set_to_product_of_lhs_is_row_matrix) {
		size_t M = 1;
		size_t K = 101;
		size_t N = 201;

		std::vector<std::vector<float>> lhs_data;
		lhs_data.resize(M);
		for (int idy = 0; idy < M; idy++) {
			for (int idx = 0; idx < K; idx++) {
				lhs_data.at(idy).push_back(9.3f * idx * idy + idx + idy + 2.3f);
			}
		}

		std::vector<std::vector<float>> rhs_data;
		rhs_data.resize(K);
		for (int idy = 0; idy < K; idy++) {
			for (int idx = 0; idx < N; idx++) {
				rhs_data.at(idy).push_back(1.2f * idx + std::log(1.f + idy));
			}
		}

		std::vector<std::vector<float>> output_data;
		output_data.resize(M);
		for (int idy = 0; idy < M; idy++) {
			for (int idx = 0; idx < N; idx++) {
				output_data.at(idy).push_back(0.0f);
			}
		}

		set_to_product_of_test_internal(output_data, lhs_data, rhs_data);
	}


	TEST(BasicOperations, set_to_product_of_lhs_is_row_matrix_and_rhs_is_wide) {
		size_t M = 1;
		size_t K = 3;
		size_t N = 3000;

		std::vector<std::vector<float>> lhs_data;
		lhs_data.resize(M);
		for (int idy = 0; idy < M; idy++) {
			for (int idx = 0; idx < K; idx++) {
				lhs_data.at(idy).push_back(9.3f * idx * idy + idx + idy + 2.3f);
			}
		}

		std::vector<std::vector<float>> rhs_data;
		rhs_data.resize(K);
		for (int idy = 0; idy < K; idy++) {
			for (int idx = 0; idx < N; idx++) {
				rhs_data.at(idy).push_back(1.2f * idx + std::log(1.f + idy));
			}
		}

		std::vector<std::vector<float>> output_data;
		output_data.resize(M);
		for (int idy = 0; idy < M; idy++) {
			for (int idx = 0; idx < N; idx++) {
				output_data.at(idy).push_back(0.0f);
			}
		}

		set_to_product_of_test_internal(output_data, lhs_data, rhs_data);
	}
}