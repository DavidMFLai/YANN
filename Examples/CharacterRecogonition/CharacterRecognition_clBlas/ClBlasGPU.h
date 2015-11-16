#pragma once

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "clBLAS.h"
#include <stdio.h>
#include <string.h>
#include <map>

class ClBlasGPU
{
public:
	ClBlasGPU();
	~ClBlasGPU();
	void matrix_multiply();
	cl_mem getClBlasBuffer(uint32_t idx, size_t size, cl_mem_flags flags);
private:
	cl_int err;
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context ctx = 0;
	cl_command_queue queue = 0;
	std::map<uint32_t, cl_mem> bufs;
};

