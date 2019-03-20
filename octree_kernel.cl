__kernel void parallel_add(__global float* x, __global float* y, __global float* z) {
	const int i = get_global_id(0);
	z[i] = y[i] + x[i];
}