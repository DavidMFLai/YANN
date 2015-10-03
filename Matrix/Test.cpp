// Matrix.cpp : Defines the entry point for the console application.
//
#include <array>
#include <tuple>

#include "gmock\gmock.h"
#include "gtest/gtest.h"
#include "Matrix.h"

TEST(BasicOperations, Default_Contruction) {
	Matrix<double> m;
	const auto dimensions = m.getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 0, 0 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, Contruction_With_Dimensions) {
	Matrix<double> m{ 4,5 };
	const auto dimensions = m.getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 4, 5 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, Contruction_With_InitializationLists) {
	Matrix<double> m{ 
		{ 1., 2. },
		{ 3., 4. },
		{ 5., 6. }
	};

	const auto dimensions = m.getDimensions();
	const std::array<size_t, 2> expectedDimensions{ 3, 2 };
	EXPECT_EQ(dimensions, expectedDimensions);
}

TEST(BasicOperations, Copy_Contruction) {
	Matrix<double> m{
		{ 1., 2. },
		{ 3., 4. },
		{ 5., 6. }
	};

	auto n{ m };

	EXPECT_EQ(m, n);
}

TEST(BasicOperations, Multiply) {
	Matrix<double> m{
		{ 111., 112. },
		{ 121., 122. },
		{ 131., 132. }
	};

	Matrix<double> n{
		{ 211., 212., 213. },
		{ 221., 222., 223.},
	};

	auto result = m*n;

	Matrix<double> expected_result{
		{ 111.*211. + 112.*221. , 111.*212. + 112.*222., 111.*213. + 112.*223. },
		{ 121.*211. + 122.*221. , 121.*212. + 122.*222., 121.*213. + 122.*223. },
		{ 131.*211. + 132.*221. , 131.*212. + 132.*222., 131.*213. + 132.*223. },
	};

	EXPECT_EQ(expected_result, result);
}

TEST(BasicOperations, Subtract) {
	Matrix<double> m{
		{ 111.1, 112.4 },
		{ 121.2, 122.5 },
		{ 131.3, 132.6 }
	};

	Matrix<double> m2{
		{ 11., 12. },
		{ 21., 22. },
		{ 31., 32. }
	};

	Matrix<double> expected_result{
		{ 100.1, 100.4 },
		{ 100.2, 100.5 },
		{ 100.3, 100.6 }
	};

	auto result = m - m2;
	
	EXPECT_EQ(expected_result, result, 0.0001);
}

TEST(BasicOperations, SubMatrix) {
	Matrix<double> m{
		{ 111., 112., 113. },
		{ 121., 122., 123. },
		{ 131., 132., 133. }
	};

	Matrix<double> expected_submatrix{
		{ 122., 123. },
		{ 132., 133. }
	};

	auto submatrix_ref = m.getSubMatrixReference(std::tuple<size_t, size_t>{ 1,2 }, std::tuple<size_t, size_t>{ 1,2 });
	auto submatrix = Matrix<double>{ submatrix_ref };

	EXPECT_EQ(expected_submatrix, submatrix);
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}