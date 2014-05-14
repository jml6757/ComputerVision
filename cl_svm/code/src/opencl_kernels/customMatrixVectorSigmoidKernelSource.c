#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void custom_matrix_vector_kernel(
											__global double * A,
											__global double * x,
											__global double * y,
											const int cols,
											__local double * scratch,
											const int localSize,
											const double gamma,
											const double coef0//,
											//const int degree
										)
{

	// variables
	int rowNumber;
	//int colNumber;
	int localIndex;
	//int localSize;
	//int numberOfWorkGroups;
	// function
	// initialize
	{ 
		// get row number 
		rowNumber = get_global_id( 0 ); 
		// get column number 
		//colNumber = get_global_id( 1 ); 
		localIndex = get_local_id( 1 ); 
		// get work group size 
		//localSize = get_local_size( 1 ); 
		//numberOfWorkGroups = get_global_size( 1 ) / localSize; 
	}
	// phase 1 reduction
	{
		int startIndex = localIndex;
		int aOffset = rowNumber * cols;
		double sum = 0.0;
		while ( startIndex < cols )
		{
			sum = sum + ( A[ aOffset + startIndex ] * x[ startIndex ] );
			startIndex = startIndex + localSize;
		}
		scratch[ localIndex ] = sum;
	}
	// BARRIER 
	{ 
		barrier( CLK_LOCAL_MEM_FENCE ); 
	} 
	// REDUCE 
	{ 
		int offset = localSize / 2; 
		// for each power of two remaining 
		while ( offset > 0 ) 
		{ 
			// if we are in the lower half 
			if ( localIndex < offset ) 
			{ 
				scratch[ localIndex ] = scratch[ localIndex ] + scratch[ localIndex + offset ]; 
			} 
			barrier( CLK_LOCAL_MEM_FENCE ); 
			// divide by two 
			offset = offset >> 1; 
		} 
	}
	// full assignment
	{
		if ( 0 == localIndex )
		{
			y[ rowNumber ] = tanh( gamma*(scratch[ 0 ]) + coef0 );
		}
	}

	// clean up
	return;
}
