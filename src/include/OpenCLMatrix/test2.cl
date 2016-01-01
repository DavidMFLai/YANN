int get_index_2D(int x, int y, int x_length) { 
	return x + y*x_length;
}

__kernel void set_to_sum_of_rows(__global float *input_and_output, __local float *partial_sums) {

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
	
	output[get_index_2D(x, y, row_length)] = multipliers[y] * multiplicand[get_index_2D(x, y, row_length)];
}

__kernel void per_column_multiply_and_then_scale(__global float *output, __global float *multipliers, __global float *multiplicand, float scale) { 
	int x = get_global_id(0);
	int y = get_global_id(1);
	int row_length = get_global_size(0);
	
	output[get_index_2D(x, y, row_length)] = multipliers[x] * multiplicand[get_index_2D(x, y, row_length)] * scale;
}

__kernel void per_column_multiply_and_then_transpose(__global float *output, __global float *multipliers, __global float *multiplicand) { 
	int x = get_global_id(0);
	int y = get_global_id(1);
	int row_length = get_global_size(0); 
	int column_length = get_global_size(1);
	
	output[get_index_2D(y, x, column_length)] = multipliers[x] * multiplicand[get_index_2D(x, y, row_length)];
}

__kernel void row_vectors_per_element_multiply_and_then_scale(__global float *output, __global float *lhs, __global float *rhs, float scale) { 
	output[get_global_id(0)] = lhs[get_global_id(0)] * rhs[get_global_id(0)] * scale;
}

__kernel void outer_product(__global float *output, __global float *lhs, __global float *rhs) { 
	int x = get_global_id(0);
	int y = get_global_id(1);
	int row_length = get_global_size(0);

	output[get_index_2D(x, y, row_length)] = lhs[y] * rhs[x];
}


__kernel void subtract_by(__global float *output, __global float *input) { 
	int x = get_global_id(0);
	int y = get_global_id(1);
	int row_length = get_global_size(0);

	output[get_index_2D(x, y, row_length)] -= input[get_index_2D(x, y, row_length)];
}

__kernel void set_to_difference_of(__global float *output, __global float *lhs, __global float *rhs) { 
	output[get_global_id(0)] = lhs[get_global_id(0)] - rhs[get_global_id(0)];
}

//very naive implementation..
__kernel void set_to_product_of(__global float *output, unsigned int M, unsigned int K, unsigned int N, __global float *lhs, __global float *rhs) { 
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	float acc = 0.f;
	for (size_t index=0; index<K; index++) {
        acc += lhs[ get_index_2D(index, y, K) ] * rhs[ get_index_2D(x, index, N) ];
    }
	output[get_index_2D(x, y, N)] = acc;
}

//optimized for a long row matrix multiplied with another matrix
__kernel void set_to_product_of_where_lhs_is_a_long_row_matrix(__global float *output, unsigned int K, unsigned int N, __global float *lhs, __global float *rhs) { 
	size_t global_id = get_global_id(0);

	float acc = 0.f;
	for (size_t index=0; index<K; index++) {
        acc += lhs[ index ] * rhs[ get_index_2D(global_id, index, N) ];
    }
	output[global_id] = acc;
}

__kernel void per_element_sigmoid(__global float *output, __global float *input) { 
	output[get_global_id(0)] = 1 / (1 + exp(-input[get_global_id(0)]));
}

__kernel void per_element_sigmoid_prime(__global float *output, __global float *input) { 
	output[get_global_id(0)] = input[get_global_id(0)] * (1 - input[get_global_id(0)]);
}

__kernel void per_element_tanh(__global float *output, __global float *input) { 
	output[get_global_id(0)] = tanh(input[get_global_id(0)]);
}

__kernel void per_element_tanh_prime(__global float *output, __global float *input) { 
	output[get_global_id(0)] = 1 - (input[get_global_id(0)] * input[get_global_id(0)]);
}