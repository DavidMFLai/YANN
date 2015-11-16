#include <array>
#include <vector>
#include <iostream>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "clBLAS.h"
#include <stdio.h>
#include <string.h>

#include "gmock\gmock.h"
#include "gtest\gtest.h"

#include "ANN.h"
#include "MinstData.h"
#include "ANNToMINSTConverter.h"

using namespace std;
using namespace CPPANN;
using namespace Converter;

/* Include CLBLAS header. It automatically includes needed OpenCL header,
* so we can drop out explicit inclusion of cl.h header.
*/


/* This example uses predefined matrices and their characteristics for
* simplicity purpose.
*/

#define M  4
#define N  3
#define K  5

static const clblasOrder order = clblasRowMajor;

static const cl_float alpha = 10;

static const clblasTranspose transA = clblasNoTrans;
static const cl_float A[M*K] = {
	11, 12, 13, 14, 15,
	21, 22, 23, 24, 25,
	31, 32, 33, 34, 35,
	41, 42, 43, 44, 45,
};
static const size_t lda = K;        /* i.e. lda = K */

static const clblasTranspose transB = clblasNoTrans;
static const cl_float B[K*N] = {
	11, 12, 13,
	21, 22, 23,
	31, 32, 33,
	41, 42, 43,
	51, 52, 53,
};
static const size_t ldb = N;        /* i.e. ldb = N */

static const cl_float beta = 20;

static cl_float C[M*N] = {
	11, 12, 13,
	21, 22, 23,
	31, 32, 33,
	41, 42, 43,
};
static const size_t ldc = N;        /* i.e. ldc = N */

static cl_float result[M*N];

static const size_t off = 1;
static const size_t offA = K + 1;   /* K + off */
static const size_t offB = N + 1;   /* N + off */
static const size_t offC = N + 1;   /* N + off */


static void
printResult(const char* str)
{
	size_t i, j, nrows;

	printf("%s:\n", str);

	nrows = (sizeof(result) / sizeof(cl_float)) / ldc;
	for (i = 0; i < nrows; i++) {
		for (j = 0; j < ldc; j++) {
			printf("%d ", (int)result[i * ldc + j]);
		}
		printf("\n");
	}
}

TEST(clBLas, basics_sgemm) 
{
	cl_int err;
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context ctx = 0;
	cl_command_queue queue = 0;
	cl_mem bufA, bufB, bufC;
	cl_event event = NULL;
	int ret = 0;

	/* Setup OpenCL environment. */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS) {
		printf("clGetPlatformIDs() failed with %d\n", err);
		FAIL();
	}

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		printf("clGetDeviceIDs() failed with %d\n", err);
		FAIL();
	}

	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("clCreateContext() failed with %d\n", err);
		FAIL();
	}

	queue = clCreateCommandQueue(ctx, device, 0, &err);
	if (err != CL_SUCCESS) {
		printf("clCreateCommandQueue() failed with %d\n", err);
		clReleaseContext(ctx);
		FAIL();
	}

	/* Setup clblas. */
	err = clblasSetup();
	if (err != CL_SUCCESS) {
		printf("clblasSetup() failed with %d\n", err);
		clReleaseCommandQueue(queue);
		clReleaseContext(ctx);
		FAIL();
	}

	/* Prepare OpenCL memory objects and place matrices inside them. */
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A),
		NULL, &err);
	bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B),
		NULL, &err);
	bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
		NULL, &err);

	err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
		M * K * sizeof(*A), A, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
		K * N * sizeof(*B), B, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
		M * N * sizeof(*C), C, 0, NULL, NULL);

	/* Call clblas extended function. Perform gemm for the lower right sub-matrices */
	err = clblasSgemm(order, transA, transB, M - off, N - off, K - off,
		alpha, bufA, offA, lda,
		bufB, offB, ldb, beta,
		bufC, offC, ldc,
		1, &queue, 0, NULL, &event);
	if (err != CL_SUCCESS) {
		printf("clblasSgemmEx() failed with %d\n", err);
		FAIL();
	}
	else {
		/* Wait for calculations to be finished. */
		err = clWaitForEvents(1, &event);

		/* Fetch results of calculations from GPU memory. */
		err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
			M * N * sizeof(*result),
			result, 0, NULL, NULL);

		/* At this point you will get the result of SGEMM placed in 'result' array. */
		puts("");
		printResult("clblasSgemmEx result");
	}

	/* Release OpenCL memory objects. */
	clReleaseMemObject(bufC);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufA);

	/* Finalize work with clblas. */
	clblasTeardown();

	/* Release OpenCL working objects. */
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);
}

TEST(CharacterRecognition, DISABLED_one_hidden_layer_with_15_neurons)
{
	string train_images_full_path = "../common/MINST/MINSTDataset/train-images.idx3-ubyte";
	string train_labels_full_path = "../common/MINST/MINSTDataset/train-labels.idx1-ubyte";
	string test_images_full_path = "../common/MINST/MINSTDataset/t10k-images.idx3-ubyte";
	string test_labels_full_path = "../common/MINST/MINSTDataset/t10k-labels.idx1-ubyte";


	//read raw training material
	MINSTData<double> mINSTData;
	mINSTData.read_data(train_images_full_path, train_labels_full_path);

	//Setup ANN
	ANNBuilder<double> ann_builder;
	auto ann = ann_builder.set_input_layer(mINSTData.get_number_of_images())
		.set_hidden_layer(0, Neuron_Type::Sigmoid, 0.5, 15)
		.set_output_layer(Neuron_Type::Sigmoid, 0.5, 10)
		.build();

	//Train with first 5000 only
	Matrix<double> training_output_data{ 1, 10 };
	for (size_t j = 0; j < 10; j++) {
		for (size_t idx = 0; idx < 5000; idx++) {
			auto &training_input_data = mINSTData.get_image(idx);
			auto training_output_data_raw = mINSTData.get_label(idx);
			Convert_label_to_ANN_output_data(training_output_data, training_output_data_raw);
			ann.forward_propagate(training_input_data);
			ann.back_propagate(training_output_data);
		}
		std::cout << "";
	}

	//read raw testing material
	MINSTData<double> mINSTData_test;
	mINSTData_test.read_data(test_images_full_path, test_labels_full_path);

	//Test
	size_t correct_count = 0;
	size_t total_count = 0;
	Matrix<double> testing_output_data{ 1, 10 };
	for (size_t idx = 0; idx < mINSTData_test.get_number_of_images(); idx++) {
		auto &test_input_data = mINSTData_test.get_image(idx);
		std::vector<double> ann_result = ann.forward_propagate(test_input_data);
		uchar result = Convert_ANN_output_data_to_label(ann_result);

		uchar label = mINSTData_test.get_label(idx);
		
		if (result == label) {
			correct_count++;
		}
		total_count++;
		
		std::cout << "Correct Ratio = " << correct_count << '/' << total_count << std::endl;

	}

}