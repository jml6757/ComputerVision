"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"#define TAU 1e-12\n" \
"\n" \
"enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };\n" \
"\n" \
"__kernel void find_candidate_j_values_kernel(  __global signed char * y,\n" \
"											   __global char * alphaStatus,\n" \
"											   __global double * QD,\n" \
"											   const int selectedI,\n" \
"											   const double gMax,\n" \
"											   const int activeSize,\n" \
"											   __global int * indexBuffer,\n" \
"											   __global double * valuesBuffer,\n" \
"											   __local int * scratchIndex,\n" \
"											   __local double * scratchValues,\n" \
"											   __global double * gMaxCandidates,\n" \
"											   __local double * scratchGMaxBuffer,\n" \
"											   __global double * Q,\n" \
"											   __global double * G\n" \
"											)\n" \
"{\n" \
"	// variables\n" \
"	int globalIndex;\n" \
"	int localIndex;\n" \
"	int localSize;\n" \
"	\n" \
"	// function body\n" \
"	// initialization\n" \
"	{\n" \
"		// get global index\n" \
"		globalIndex = get_global_id( 0 );\n" \
"		// get local index\n" \
"		localIndex = get_local_id( 0 );\n" \
"		// get work group size\n" \
"		localSize = get_local_size( 0 );\n" \
"		// scratch index <-- global index\n" \
"		scratchIndex[ localIndex ] = globalIndex;\n" \
"		// scratch gmax <-- -INF\n" \
"		scratchGMaxBuffer[ localIndex ] = -DBL_MAX;\n" \
"	}\n" \
"	// independent calculation\n" \
"	{\n" \
"		double gradDiff;\n" \
"		double quadCoefficient;\n" \
"		// scratch values <-- INF\n" \
"		scratchValues[ localIndex ] = DBL_MAX;\n" \
"		// if global index < active size AND y is positive AND NOT lower bound\n" \
"		if ( globalIndex < activeSize ) \n" \
"		{\n" \
"			if ( +1 == y[ globalIndex ] && LOWER_BOUND != alphaStatus[ globalIndex ] )\n" \
"			{\n" \
"				// scratch g max <-- G\n" \
"				scratchGMaxBuffer[ localIndex ] = G[ globalIndex ];\n" \
"				// calculate grad diff\n" \
"				gradDiff = gMax + G[ globalIndex ];\n" \
"				// if grad diff > 0\n" \
"				if ( gradDiff > 0 )\n" \
"				{\n" \
"					// calculate quad coefficient\n" \
"					quadCoefficient = QD[ selectedI ] + QD[ globalIndex ] - ( 2.0 * y[ selectedI ] * Q[ globalIndex ]);\n" \
"					// if quad coefficient > 0\n" \
"					if ( quadCoefficient > 0 )\n" \
"					{\n" \
"						// scratch values <-- -(grad_diff^2)/quad_coef\n" \
"						scratchValues[ localIndex ] = -( gradDiff * gradDiff )/quadCoefficient;\n" \
"					}\n" \
"					// otherwise\n" \
"					else\n" \
"					{\n" \
"						// scratch values <-- -(grad_diff^2)/TAU\n" \
"						scratchValues[ localIndex ] = -( gradDiff * gradDiff )/((double)TAU);\n" \
"					}\n" \
"				}\n" \
"			}\n" \
"		}\n" \
"		// if global index < active size AND y is NOT positive AND NOT upper bound\n" \
"		if ( globalIndex < activeSize )\n" \
"		{\n" \
"			if ( +1 != y[ globalIndex ] && UPPER_BOUND != alphaStatus[ globalIndex ] )\n" \
"			{\n" \
"				// scratch g max <-- -G\n" \
"				scratchGMaxBuffer[ localIndex ] = -G[ globalIndex ] ;\n" \
"				// calculate grad diff\n" \
"				gradDiff = gMax - G[ globalIndex ];\n" \
"				// if grad diff > 0\n" \
"				if ( gradDiff > 0 )\n" \
"				{\n" \
"					// calculate quad coeff\n" \
"					quadCoefficient = QD[ selectedI ] + QD[ globalIndex ] + ( 2.0 * y[ selectedI ] * Q[ globalIndex ]);\n" \
"					// if quad coeff > 0\n" \
"					if ( quadCoefficient > 0 )\n" \
"					{\n" \
"						// scratch values <-- -(grad_diff^2)/quad_coef\n" \
"						scratchValues[ localIndex ] = -( gradDiff * gradDiff )/quadCoefficient;\n" \
"					}\n" \
"					// otherwise\n" \
"					else\n" \
"					{\n" \
"						// scratch values <-- -(grad_diff^2)/TAU\n" \
"						scratchValues[ localIndex ] = -( gradDiff * gradDiff )/TAU;\n" \
"					}\n" \
"				}\n" \
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
"				// if we are not the min\n" \
"				if ( scratchValues[ localIndex ] > scratchValues[ localIndex + offset ] )\n" \
"				{\n" \
"					// scratch value <-- min\n" \
"					scratchValues[ localIndex ] = scratchValues[ localIndex + offset ];\n" \
"					// scratch index <-- min index\n" \
"					scratchIndex[ localIndex ] = scratchIndex[ localIndex + offset ];\n" \
"				}\n" \
"				// if we don't have the G MAX\n" \
"				if ( scratchGMaxBuffer[ localIndex ] < scratchGMaxBuffer[ localIndex + offset ] )\n" \
"				{\n" \
"					// scratch gmax <-- max\n" \
"					scratchGMaxBuffer[ localIndex ] = scratchGMaxBuffer[ localIndex + offset ];\n" \
"				}\n" \
"			}\n" \
"			barrier( CLK_LOCAL_MEM_FENCE );\n" \
"			// divide by two\n" \
"			offset = offset >> 1;\n" \
"		}\n" \
"	}\n" \
"	// barrier\n" \
"	// output assignment\n" \
"	{\n" \
"		int outputIndex = globalIndex / localSize;\n" \
"		// if we are the zeroth index\n" \
"		if ( 0 == localIndex )\n" \
"		{\n" \
"			// output value <-- scratch value\n" \
"			valuesBuffer[ outputIndex ] = scratchValues[ 0 ];\n" \
"			// output index <-- scratch index\n" \
"			indexBuffer[ outputIndex ] = scratchIndex[ 0 ];\n" \
"			// output gmax <-- scratch gmax\n" \
"			gMaxCandidates[ outputIndex ] = scratchGMaxBuffer[ 0 ];\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
