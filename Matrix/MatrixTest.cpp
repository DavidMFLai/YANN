// Matrix.cpp : Defines the entry point for the console application.
//
#include <array>
#include <tuple>

#include "gmock\gmock.h"
#include "gtest/gtest.h"
#include "ReferenceMatrix.h"

TEST(BasicOperations, Default_Contruction) {
	ReferenceMatrix<double> m;
	const auto dimensions = m.getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 0, 0 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, Contruction_With_Dimensions) {
	ReferenceMatrix<double> m{ 4,5 };
	const auto dimensions = m.getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 4, 5 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, Contruction_With_InitializationLists) {
	ReferenceMatrix<double> m{
		{ 1., 2. },
		{ 3., 4. },
		{ 5., 6. }
	};

	const auto dimensions = m.getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 3, 2 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, Copy_Contruction) {
	ReferenceMatrix<double> m{
		{ 1., 2. },
		{ 3., 4. },
		{ 5., 6. }
	};

	auto n{ m };

	EXPECT_EQ(m, n);
}

TEST(BasicOperations, FunctionMultiplyDefault) {
	ReferenceMatrix<double> m{
		{ 111., 112. },
		{ 121., 122. },
		{ 131., 132. }
	};

	ReferenceMatrix<double> n{
		{ 211., 212., 213. },
		{ 221., 222., 223. },
	};

	ReferenceMatrix<double> result(3, 3);

	ReferenceMatrix<double>::Multiply(result, m, n);

	ReferenceMatrix<double> expected_result{
		{ 111.*211. + 112.*221. , 111.*212. + 112.*222., 111.*213. + 112.*223. },
		{ 121.*211. + 122.*221. , 121.*212. + 122.*222., 121.*213. + 122.*223. },
		{ 131.*211. + 132.*221. , 131.*212. + 132.*222., 131.*213. + 132.*223. },
	};

	EXPECT_EQ(expected_result, result);
}

TEST(BasicOperations, sumOfRows) {
	ReferenceMatrix<double> input{
		{ 1., 2., 3. },
		{ 4., 5., 6. },
	};

	ReferenceMatrix<double> expected{
		{ 1. + 4., 2. + 5., 3. + 6. },
	};

	ReferenceMatrix<double> result{1, 3};
	ReferenceMatrix<double>::Sum_of_rows(result, input);

	EXPECT_EQ(expected, result);
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}