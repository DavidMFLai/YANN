// Matrix.cpp : Defines the entry point for the console application.
//
#include <array>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ReferenceMatrix.h"
#include "ReferenceMatrixBuilder.h"

using namespace YANN;

TEST(BasicOperations, Contruction_With_Dimensions) {
	auto m = ReferenceMatrixBuilder<double>{}.build(4, 5);
	const auto dimensions = m->getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 4, 5 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, Contruction_With_InitializationLists) {
	auto m = ReferenceMatrixBuilder<double>{}.build({
		{ 1., 2. },
		{ 3., 4. },
		{ 5., 6. }
	});
 	const auto dimensions = m->getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 3, 2 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, FunctionMultiplyDefault) {
	auto builder = ReferenceMatrixBuilder<double>{};

	auto m = builder.build({
		{ 111., 112. },
		{ 121., 122. },
		{ 131., 132. }
	});

	auto n = builder.build({
		{ 211., 212., 213. },
		{ 221., 222., 223. },
	});

	auto result = builder.build( 3, 3 );

	ReferenceMatrix<double>::Multiply(*result, *m, *n);

	auto expected = builder.build({
		{ 111.*211. + 112.*221. , 111.*212. + 112.*222., 111.*213. + 112.*223. },
		{ 121.*211. + 122.*221. , 121.*212. + 122.*222., 121.*213. + 122.*223. },
		{ 131.*211. + 132.*221. , 131.*212. + 132.*222., 131.*213. + 132.*223. },
	});

	EXPECT_EQ(*expected, *result);
}

TEST(BasicOperations, sumOfRows) {
	auto builder = ReferenceMatrixBuilder<double>{};

	auto input = builder.build({
		{ 1., 2., 3. },
		{ 4., 5., 6. },
	});

	auto expected = builder.build({
		{ 1. + 4., 2. + 5., 3. + 6. }
	});

	auto result = builder.build( 1, 3 );

	ReferenceMatrix<double>::Sum_of_rows(*result, *input);

	EXPECT_EQ(*expected, *result);
}

TEST(ReferenceMatrixBuilder, buildBySettingDimensions) {
	ReferenceMatrixBuilder<double> ref_matrix_builder;
	auto created_matrix = ref_matrix_builder.build( 1, 2 );

	EXPECT_EQ(1, created_matrix->getColumnLength());
	EXPECT_EQ(2, created_matrix->getRowLength());
}

TEST(ReferenceMatrixBuilder, buildBySettingValues) {
	ReferenceMatrixBuilder<double> ref_matrix_builder;
	auto created_matrix = ref_matrix_builder.build({ 
		{ 1, 2, 3 },
		{ 4, 5, 6 } 
	});

	EXPECT_EQ(2, created_matrix->getColumnLength());
	EXPECT_EQ(3, created_matrix->getRowLength());
}