__kernel void add(__global float *a,
                  __global float *b,
                  __global float *c) {
   
   *c = *a + *b;
}

__kernel void sub(__global float *a,
                  __global float *b,
                  __global float *c) {
   
   *c = *a - *b;
}

__kernel void mult(__global float *a,
                   __global float *b,
                   __global float *c) {
   
   *c = *a * *b;
}

__kernel void div(__global float *a,
                  __global float *b,
                  __global float *c) {
   
   *c = *a / *b;
}

__kernel void id_check(__global float *output) {
	
	size_t global_id_0 = get_global_id(0);
	size_t global_id_1 = get_global_id(1);
	size_t global_size_0 = get_global_size(0);
	size_t offset_0 = get_global_offset(0);
	size_t offset_1 = get_global_offset(1);
	size_t local_id_0 = get_local_id(0);
	size_t local_id_1 = get_local_id(1);

	int index_0 = global_id_0 - offset_0;
	int index_1 = global_id_1 - offset_1;
	int index = index_1 * global_size_0 + index_0;

	float f = global_id_0 * 10.f + global_id_1 * 1.f;
	f += local_id_0 * 0.1f + local_id_1 * 0.01f;

	output[index] = f;

}

__kernel void reduction_scalar(__global float* data,  __local float* partial_sums, __global float* output) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_sums[lid] = data[get_global_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      output[get_group_id(0)] = partial_sums[0];
   }
}

__kernel void sum(__global float *output, __local float *partial_sums, __global float *input) {

   int lid = get_local_id(1);
   int group_size = get_local_size(1);

   partial_sums[lid] = input[get_global_id(0) + get_global_size(0) * get_global_id(1)];
   barrier(CLK_LOCAL_MEM_FENCE);
  
  
   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   
   if(lid == 0) {
      output[get_group_id(0) + get_num_groups(0) * get_group_id(1)] = partial_sums[0];
   }
}