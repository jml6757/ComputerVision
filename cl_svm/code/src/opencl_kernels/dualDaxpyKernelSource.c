#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void dual_daxpy_kernel(
									__global double * G,
									//__global double * Q1,
									__global float * Q1,
									//__global double * Q2,
									__global float * Q2,
									const double alpha1,
									//const float alpha1,
									const double alpha2,
									//const float alpha2,
									const int activeSize
								)
{
	// variables
	int globalIndex = get_global_id( 0 );
	
	// function
	// REALLY straightforward
	//if ( (globalIndex = get_global_id( 0 )) < activeSize )
	//while ( globalIndex < activeSize )
	if ( globalIndex < activeSize )
	{
		G[ globalIndex ] += alpha1 * Q1[ globalIndex ] + alpha2 * Q2[ globalIndex ];
		//globalIndex += groupSize;
	}
	
	// clean up
	return;
}
