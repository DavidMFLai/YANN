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
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
