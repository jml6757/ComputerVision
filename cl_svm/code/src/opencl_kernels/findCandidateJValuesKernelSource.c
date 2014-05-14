#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define TAU 1e-12

enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };

__kernel void find_candidate_j_values_kernel(  __global signed char * y,
											   __global char * alphaStatus,
											   __global double * QD,
											   const int selectedI,
											   const double gMax,
											   const int activeSize,
											   __global int * indexBuffer,
											   __global double * valuesBuffer,
											   __local int * scratchIndex,
											   __local double * scratchValues,
											   __global double * gMaxCandidates,
											   __local double * scratchGMaxBuffer,
											   __global double * Q,
											   __global double * G
											)
{
	// variables
	int globalIndex;
	int localIndex;
	int localSize;
	
	// function body
	// initialization
	{
		// get global index
		globalIndex = get_global_id( 0 );
		// get local index
		localIndex = get_local_id( 0 );
		// get work group size
		localSize = get_local_size( 0 );
		// scratch index <-- global index
		scratchIndex[ localIndex ] = globalIndex;
		// scratch gmax <-- -INF
		scratchGMaxBuffer[ localIndex ] = -DBL_MAX;
	}
	// independent calculation
	{
		double gradDiff;
		double quadCoefficient;
		// scratch values <-- INF
		scratchValues[ localIndex ] = DBL_MAX;
		// if global index < active size AND y is positive AND NOT lower bound
		if ( globalIndex < activeSize ) 
		{
			if ( +1 == y[ globalIndex ] && LOWER_BOUND != alphaStatus[ globalIndex ] )
			{
				// scratch g max <-- G
				scratchGMaxBuffer[ localIndex ] = G[ globalIndex ];
				// calculate grad diff
				gradDiff = gMax + G[ globalIndex ];
				// if grad diff > 0
				if ( gradDiff > 0 )
				{
					// calculate quad coefficient
					quadCoefficient = QD[ selectedI ] + QD[ globalIndex ] - ( 2.0 * y[ selectedI ] * Q[ globalIndex ]);
					// if quad coefficient > 0
					if ( quadCoefficient > 0 )
					{
						// scratch values <-- -(grad_diff^2)/quad_coef
						scratchValues[ localIndex ] = -( gradDiff * gradDiff )/quadCoefficient;
					}
					// otherwise
					else
					{
						// scratch values <-- -(grad_diff^2)/TAU
						scratchValues[ localIndex ] = -( gradDiff * gradDiff )/((double)TAU);
					}
				}
			}
		}
		// if global index < active size AND y is NOT positive AND NOT upper bound
		if ( globalIndex < activeSize )
		{
			if ( +1 != y[ globalIndex ] && UPPER_BOUND != alphaStatus[ globalIndex ] )
			{
				// scratch g max <-- -G
				scratchGMaxBuffer[ localIndex ] = -G[ globalIndex ] ;
				// calculate grad diff
				gradDiff = gMax - G[ globalIndex ];
				// if grad diff > 0
				if ( gradDiff > 0 )
				{
					// calculate quad coeff
					quadCoefficient = QD[ selectedI ] + QD[ globalIndex ] + ( 2.0 * y[ selectedI ] * Q[ globalIndex ]);
					// if quad coeff > 0
					if ( quadCoefficient > 0 )
					{
						// scratch values <-- -(grad_diff^2)/quad_coef
						scratchValues[ localIndex ] = -( gradDiff * gradDiff )/quadCoefficient;
					}
					// otherwise
					else
					{
						// scratch values <-- -(grad_diff^2)/TAU
						scratchValues[ localIndex ] = -( gradDiff * gradDiff )/TAU;
					}
				}
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
				// if we are not the min
				if ( scratchValues[ localIndex ] > scratchValues[ localIndex + offset ] )
				{
					// scratch value <-- min
					scratchValues[ localIndex ] = scratchValues[ localIndex + offset ];
					// scratch index <-- min index
					scratchIndex[ localIndex ] = scratchIndex[ localIndex + offset ];
				}
				// if we don't have the G MAX
				if ( scratchGMaxBuffer[ localIndex ] < scratchGMaxBuffer[ localIndex + offset ] )
				{
					// scratch gmax <-- max
					scratchGMaxBuffer[ localIndex ] = scratchGMaxBuffer[ localIndex + offset ];
				}
			}
			barrier( CLK_LOCAL_MEM_FENCE );
			// divide by two
			offset = offset >> 1;
		}
	}
	// barrier
	// output assignment
	{
		int outputIndex = globalIndex / localSize;
		// if we are the zeroth index
		if ( 0 == localIndex )
		{
			// output value <-- scratch value
			valuesBuffer[ outputIndex ] = scratchValues[ 0 ];
			// output index <-- scratch index
			indexBuffer[ outputIndex ] = scratchIndex[ 0 ];
			// output gmax <-- scratch gmax
			gMaxCandidates[ outputIndex ] = scratchGMaxBuffer[ 0 ];
		}
	}
	
	// clean up
	return;
}
