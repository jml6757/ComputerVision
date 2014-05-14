#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void custom_daxpy_kernel(
									__global double * x,
									const double c
								 )
{
	// variables
	int localIndex;
	
	// function body
	// get the index we're working on
	localIndex = get_local_id( 0 );
	// output[ index ] = input[ index ] + c
	x[ localIndex ] = x[ localIndex ] + c;
	
	// clean up
	return;
}
