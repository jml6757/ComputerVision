"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"\n" \
"enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };\n" \
"\n" \
"__kernel void find_candidate_i_values_kernel( __global double * G,\n" \
"											  __global char * y,\n" \
"											  __global signed char * alphaStatus,\n" \
"											  const int activeSize,\n" \
"											  __global int * indexBuffer,\n" \
"											  __global double * valueBuffer,\n" \
"											  __local int * scratchIndexBuffer,\n" \
"											  __local double * scratchValueBuffer\n" \
"											)\n" \
"{\n" \
"	// variables\n" \
"	int localIndex;\n" \
"	int globalIndex;\n" \
"	int localSize;\n" \
"	\n" \
"	// function body\n" \
"	// initialization\n" \
"	{\n" \
"		// get local index\n" \
"		localIndex = get_local_id( 0 );\n" \
"		// get global index\n" \
"		globalIndex = get_global_id( 0 );\n" \
"		// get work group size\n" \
"		localSize = get_local_size( 0 );\n" \
"		// saved local index\n" \
"		scratchIndexBuffer[ localIndex ] = globalIndex;\n" \
"		//scratchIndexBuffer[ globalIndex ] = globalIndex;\n" \
"	}\n" \
"	// initial assignment\n" \
"	{\n" \
"		scratchValueBuffer[ localIndex ] = -DBL_MAX;\n" \
"		// if global index > active size\n" \
"		if ( globalIndex >= activeSize )\n" \
"		{\n" \
"			// scratch value <-- -INF\n" \
"			scratchValueBuffer[ localIndex ] = -DBL_MAX;\n" \
"		}\n" \
"		// otherwise\n" \
"		else\n" \
"		{\n" \
"			// if positive label and NOT upper bound\n" \
"			if ( +1 == y[ globalIndex ] && UPPER_BOUND != alphaStatus[ globalIndex ] )\n" \
"			{\n" \
"				// scratch value <-- -G\n" \
"				scratchValueBuffer[ localIndex ] = -G[ globalIndex ];\n" \
"			}\n" \
"			// if NOT positive label and NOT lower bound\n" \
"			if ( +1 != y[ globalIndex ] && LOWER_BOUND != alphaStatus[ globalIndex ] )\n" \
"			{\n" \
"				// scratch value <-- G\n" \
"				scratchValueBuffer[ localIndex ] = G[ globalIndex ];\n" \
"			}\n" \
"		}\n" \
"	}\n" \
"	// barrier\n" \
"	{\n" \
"		barrier( CLK_LOCAL_MEM_FENCE );\n" \
"	}\n" \
"	// reduction\n" \
"	{\n" \
"		// get local work group size\n" \
"		int offset = localSize / 2;\n" \
"		// for each power of two remaining\n" \
"		while ( offset > 0 )\n" \
"		{\n" \
"			// if we are in the lower half\n" \
"			if ( localIndex < offset )\n" \
"			{\n" \
"				// if we are not the max\n" \
"				if ( scratchValueBuffer[ localIndex ] < scratchValueBuffer[ localIndex + offset ] )\n" \
"				{\n" \
"					// scratch value <-- max\n" \
"					scratchValueBuffer[ localIndex ] = scratchValueBuffer[ localIndex + offset ];\n" \
"					// scratch index <-- max index\n" \
"					scratchIndexBuffer[ localIndex ] = scratchIndexBuffer[ localIndex + offset ];\n" \
"				}\n" \
"			}\n" \
"			barrier( CLK_LOCAL_MEM_FENCE );\n" \
"			// divide by two\n" \
"			offset = offset >> 1;\n" \
"		}\n" \
"	}\n" \
"	// write out\n" \
"	{\n" \
"		int outputIndex = globalIndex / localSize;\n" \
"		// if we are the zeroth index\n" \
"		if ( 0 == localIndex )\n" \
"		{\n" \
"			// write our value to the global output according to our block number\n" \
"			indexBuffer[ outputIndex ] = scratchIndexBuffer[ 0 ];\n" \
"			valueBuffer[ outputIndex ] = scratchValueBuffer[ 0 ];\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
