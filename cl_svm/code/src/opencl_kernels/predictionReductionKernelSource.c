#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void prediction_reduction_kernel(
											__global double * y,
											__global double * alpha,
											__local double * scratch,
											__global double * sum,
											const int cols,
											const int workGroupSize
										 )
{
	// variables
	int localIndex;
	double localSum;
	
	// function body
	// initialization
	{
		// get local index
		localIndex = get_local_id( 0 );
	}
	// reduction, phase 1
	{
		int i;
		localSum = 0.0;
		// for i = local index; i < cols; i += workGroupSize
		for ( i = localIndex; i < cols; i = i + workGroupSize )
		{
			// sum += y[ i ] * alpha[ i ]
			localSum += y[ i ] * alpha[ i ];
		}
		// scratch[ local index ] = sum
		scratch[ localIndex ] = localSum;
	}
	barrier( CLK_LOCAL_MEM_FENCE );
	// reduction, phase 2 (actually in parallel)
	{
		int offset;
		// for offset = workGroupSize / 2; offset > 0; offset >>= 1
		for ( offset = workGroupSize / 2; offset > 0; offset = offset >> 1 )
		{
			// if local index < offset
			if ( localIndex < offset )
			{
				// scratch[ local index ] += scratch[ local index + offset ]
				scratch[ localIndex ] += scratch[ localIndex + offset ];
			}
			barrier( CLK_LOCAL_MEM_FENCE );
		}
	}
	// final assignment
	{
		// if 0 == local index
		if ( 0 == localIndex )
		{
			// sum[ 0 ] = scratch[ 0 ]
			sum[ 0 ] = scratch[ 0 ];
		}
	}
	
	// clean up
	return;
}

