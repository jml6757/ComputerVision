#pragma OPENCL EXTENSION cl_khr_fp64 : enable

enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };

__kernel void find_candidate_i_values_kernel( __global double * G,
											  __global char * y,
											  __global signed char * alphaStatus,
											  const int activeSize,
											  __global int * indexBuffer,
											  __global double * valueBuffer,
											  __local int * scratchIndexBuffer,
											  __local double * scratchValueBuffer
											)
{
	// variables
	int localIndex;
	int globalIndex;
	int localSize;
	
	// function body
	// initialization
	{
		// get local index
		localIndex = get_local_id( 0 );
		// get global index
		globalIndex = get_global_id( 0 );
		// get work group size
		localSize = get_local_size( 0 );
		// saved local index
		scratchIndexBuffer[ localIndex ] = globalIndex;
		//scratchIndexBuffer[ globalIndex ] = globalIndex;
	}
	// initial assignment
	{
		scratchValueBuffer[ localIndex ] = -DBL_MAX;
		// if global index > active size
		if ( globalIndex >= activeSize )
		{
			// scratch value <-- -INF
			scratchValueBuffer[ localIndex ] = -DBL_MAX;
		}
		// otherwise
		else
		{
			// if positive label and NOT upper bound
			if ( +1 == y[ globalIndex ] && UPPER_BOUND != alphaStatus[ globalIndex ] )
			{
				// scratch value <-- -G
				scratchValueBuffer[ localIndex ] = -G[ globalIndex ];
			}
			// if NOT positive label and NOT lower bound
			if ( +1 != y[ globalIndex ] && LOWER_BOUND != alphaStatus[ globalIndex ] )
			{
				// scratch value <-- G
				scratchValueBuffer[ localIndex ] = G[ globalIndex ];
			}
		}
	}
	// barrier
	{
		barrier( CLK_LOCAL_MEM_FENCE );
	}
	// reduction
	{
		// get local work group size
		int offset = localSize / 2;
		// for each power of two remaining
		while ( offset > 0 )
		{
			// if we are in the lower half
			if ( localIndex < offset )
			{
				// if we are not the max
				if ( scratchValueBuffer[ localIndex ] < scratchValueBuffer[ localIndex + offset ] )
				{
					// scratch value <-- max
					scratchValueBuffer[ localIndex ] = scratchValueBuffer[ localIndex + offset ];
					// scratch index <-- max index
					scratchIndexBuffer[ localIndex ] = scratchIndexBuffer[ localIndex + offset ];
				}
			}
			barrier( CLK_LOCAL_MEM_FENCE );
			// divide by two
			offset = offset >> 1;
		}
	}
	// write out
	{
		int outputIndex = globalIndex / localSize;
		// if we are the zeroth index
		if ( 0 == localIndex )
		{
			// write our value to the global output according to our block number
			indexBuffer[ outputIndex ] = scratchIndexBuffer[ 0 ];
			valueBuffer[ outputIndex ] = scratchValueBuffer[ 0 ];
		}
	}
	
	// clean up
	return;
}
