#include "ClBlasGPU.h"
#include <vector>
#include <string>
#include <sstream>

template <typename T>
std::string number_to_string(T Number)
{
	std::stringstream ss;
	ss << Number;
	return ss.str();
}

ClBlasGPU::ClBlasGPU()
{
	int ret = 0;

	/* Setup OpenCL environment. */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS) {
		printf("clGetPlatformIDs() failed with %d\n", err);
		throw std::runtime_error(number_to_string(err));
	}

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		printf("clGetDeviceIDs() failed with %d\n", err);
		throw std::runtime_error(number_to_string(err));
	}

	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("clCreateContext() failed with %d\n", err);
		throw std::runtime_error(number_to_string(err));
	}

	queue = clCreateCommandQueue(ctx, device, 0, &err);
	if (err != CL_SUCCESS) {
		printf("clCreateCommandQueue() failed with %d\n", err);
		clReleaseContext(ctx);
		throw std::runtime_error(number_to_string(err));
	}

	/* Setup clblas. */
	err = clblasSetup();
	if (err != CL_SUCCESS) {
		printf("clblasSetup() failed with %d\n", err);
		clReleaseCommandQueue(queue);
		clReleaseContext(ctx);
		throw std::runtime_error(number_to_string(err));
	}
}


ClBlasGPU::~ClBlasGPU()
{
	/* Release OpenCL memory objects. */
	for (auto buf : this->bufs) {
		clReleaseMemObject(buf.second);
	}

	/* Finalize work with clblas. */
	clblasTeardown();

	/* Release OpenCL working objects. */
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);
}

void ClBlasGPU::matrix_multiply() {
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
		std::string errMsg = "clblasSgemmEx() failed with " + number_to_string(err);
		throw std::runtime_error(errMsg);
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
}

cl_mem ClBlasGPU::getClBlasBuffer(uint32_t idx, size_t size_in_bytes, cl_mem_flags flags)
{
	auto search = this->bufs.find(idx);
	if (search != this->bufs.end()) {
		return search->second;
	}
	else {
		//create a new OpenCL memory object 
		cl_int err;
		cl_mem buf = clCreateBuffer(ctx, flags, size_in_bytes, NULL, &err);
		if (err != 0) {
			throw std::runtime_error(number_to_string(err));
		}

		//put that into bufs
		this->bufs.insert({ idx, buf });
		return buf;
	}
}
