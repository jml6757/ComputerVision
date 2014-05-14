#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf 

__kernel void linear_kernel_kernel(
									__global double * x_data_i,
									__global double * x_data_j,
									__global double * outputData,
									int goodDataSize,
									__local double * scratchData
								  )
{
	// variables
	int localIndex;
	int offset;
	
	// function
	// get the index that we will work on
	localIndex = get_local_id( 0 );
	// perform our simple multiplication (this is a dot product, after all )
	//if ( localIndex < goodDataSize )
	{
		//scratchData[ localIndex ] = x_data_i[ localIndex ] * x_data_j[ localIndex ];
		outputData[ localIndex ] = x_data_i[ localIndex ] * x_data_j[ localIndex ];
	}
	/*else
	{
		// additive identity for padding
		scratchData[ localIndex ] = 0.0;
	}
	// REDUCTION
	barrier( CLK_LOCAL_MEM_FENCE );
	// for each round
	for ( offset = get_local_size(0) / 2; offset > 0; offset = offset >> 1 )
	{
		// if we are in this round
		if ( localIndex < offset )
		{
			// add
			scratchData[ localIndex ] = scratchData[ localIndex ] + scratchData[ localIndex + offset ];
		}
		// fence
		barrier( CLK_LOCAL_MEM_FENCE );
	}
	// END OF REDUCTION
	// assign output
	if ( 0 == localIndex )
	{
		outputData[ 0 ] = scratchData[ 0 ];
		//outputData[ 0 ] = 13.0;
	}*/
	
	// clean up
	return;
}
