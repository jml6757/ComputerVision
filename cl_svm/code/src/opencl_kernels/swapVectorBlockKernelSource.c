#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void swap_vector_block_kernel(
										__global double * A,
										const int rows,
										const int cols,
										const int i,
										const int j
									 )
{
	// variables
	int globalIndex;
	int matrixOffset1;
	int matrixOffset2;
	
	// function
	// figure out which index we're working on
	{
		// get our global id
		globalIndex = get_global_id( 0 );
		// calculate our offsets into the matrix
		matrixOffset1 = i * cols + globalIndex;
		matrixOffset2 = j * cols + globalIndex;
	}
	// swap two elements
	{
		double element1;
		double element2;
		// if our column is in the matrix
		if ( globalIndex < cols )
		{
			// load the element we're responsible for from the first row
			element1 = A[ matrixOffset1 ];
			// load the element from the second row
			element2 = A[ matrixOffset2 ];
			// write them both back to the opposite rows
			A[ matrixOffset1 ] = element2;
			A[ matrixOffset2 ] = element1;
		}
	}
	
	// clean up
	return;
}
