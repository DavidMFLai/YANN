int get_index_2D(int x, int y, int x_length) { 
	return x + y*x_length;
}

__kernel void used_by_set_to_sum_of_rows(__global float *input_and_output, __local float *partial_sums) {

	int lid = get_local_id(1);
	int current_size = get_local_size(1);

	partial_sums[lid] = input_and_output[get_global_id(0) + get_global_size(0) * get_global_id(1)];
	barrier(CLK_LOCAL_MEM_FENCE);

	while(current_size > 1) {
		int addition_offset = current_size/2;
		if(lid < addition_offset) {
			partial_sums[lid] += partial_sums[lid + addition_offset];
		}
		if(current_size & 0x1 == 0x1) {
			if(lid == addition_offset) {
				partial_sums[lid] = partial_sums[current_size - 1];
			}
			current_size = addition_offset + 1;
		}
		else {
			current_size = addition_offset;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

   if(lid == 0) {
      input_and_output[get_group_id(0) + get_num_groups(0) * get_group_id(1)] = partial_sums[0];
   }
}

__kernel void set_to_sum_of(__global float *output, __global float *lhs, __global float *rhs) { 
	output[get_global_id(0)] = lhs[get_global_id(0)] + rhs[get_global_id(0)];
}

__kernel void per_row_multiply_reduction(__global float *output, __global float *multipliers, __global float *multiplicand) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int row_length = get_global_size(0);
	
	output[get_index_2D(x, y, row_length)] = multipliers[x] * multiplicand[get_index_2D(x, y, row_length)];
}