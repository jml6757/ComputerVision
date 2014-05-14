#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void reduction_kernel(
								__global double * unreducedData,
								const int cols,
								__global double * reducedData
							  )
{

	// variables
	int globalIndex;
	int offset;
	double sum;

	// function
	// initialization
	{
		// get global index
		globalIndex = get_global_id( 0 );
		// get row based offset
		offset = globalIndex * cols;
	}
	// add all the data together
	{
		sum = 0.0;
		for ( int i = 0; i < cols; i++ )
		{
			sum += unreducedData[ offset + i ];
		}
	}
	// assign output
	{
		reducedData[ globalIndex ] = sum;
	}

	// clean up
	return;
}
