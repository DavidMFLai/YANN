// OpenCLMatrix.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "gmock\gmock.h"
#include "gtest/gtest.h"

#include "Matrix.h"
#include "include\OpenCLMatrix.h"
#include "include\OpenCLMatrixBuilder.h"

TEST(BasicOperations, Default_Contruction) {
	
	OpenCLMatrixBuilder<float> builder;
	auto m = builder.create(3, 4);

	const auto dimensions = m->getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 3, 4 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
