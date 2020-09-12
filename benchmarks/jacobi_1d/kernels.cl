
__kernel void jacobi_1d_df_right(__global double* center_in, __global double* right_in, __global double* center_out, __global double* right_out)
{
	int x = get_global_id(0);
	int block_size_x = get_global_size(0);

	int index_center = 0 + x;
	int index_right_center = 0 + (x+1);

	int index_right = 0;

	double center_val = center_in[index_center];
	double right_val = (x == block_size_x-1) ? right_in[index_right] : center_in[index_right_center];

	double new_val = (center_val+right_val) / 3.0;
	center_out[index_center] = new_val;

	if(x == block_size_x-1)
		right_out[index_right] = new_val;
}

__kernel void jacobi_1d_df_left_right(__global double* center_in, __global double* left_in, __global double* right_in, __global double* center_out, __global double* left_out, __global double* right_out)
{
	int x = get_global_id(0);
	int block_size_x = get_global_size(0);

	int index_center = 0 + x;
	int index_left_center = 0 + (x-1);
	int index_right_center = 0 + (x+1);

	int index_left = 0;
	int index_right = 0;

	double center_val = center_in[index_center];
	double left_val = (x == 0) ? left_in[index_left] : center_in[index_left_center];
	double right_val = (x == block_size_x-1) ? right_in[index_right] : center_in[index_right_center];

	double new_val = (center_val+left_val+right_val) / 3.0;
	center_out[index_center] = new_val;

	if(x == 0)
		left_out[index_left] = new_val;

	if(x == block_size_x-1)
		right_out[index_right] = new_val;
}

__kernel void jacobi_1d_df_left(__global double* center_in, __global double* left_in, __global double* center_out, __global double* left_out)
{
	int x = get_global_id(0);
	int block_size_x = get_global_size(0);

	int index_center = 0 + x;
	int index_left_center = 0 + (x-1);

	int index_left = 0;

	double center_val = center_in[index_center];
	double left_val = (x == 0) ? left_in[index_left] : center_in[index_left_center];

	double new_val = (center_val+left_val) / 3.0;
	center_out[index_center] = new_val;

	if(x == 0)
		left_out[index_left] = new_val;
}

__kernel void jacobi_1d_df(__global double* center_in, __global double* center_out)
{
	int x = get_global_id(0);
	int block_size_x = get_global_size(0);

	int index_center = 0 + x;


	double center_val = center_in[index_center];

	double new_val = (center_val) / 3.0;
	center_out[index_center] = new_val;
}


