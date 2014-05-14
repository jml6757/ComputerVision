"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"\n" \
"__kernel void prediction_reduction_kernel(\n" \
"											__global double * y,\n" \
"											__global double * alpha,\n" \
"											__local double * scratch,\n" \
"											__global double * sum,\n" \
"											const int cols,\n" \
"											const int workGroupSize\n" \
"										 )\n" \
"{\n" \
"	// variables\n" \
"	int localIndex;\n" \
"	double localSum;\n" \
"	\n" \
"	// function body\n" \
"	// initialization\n" \
"	{\n" \
"		// get local index\n" \
"		localIndex = get_local_id( 0 );\n" \
"	}\n" \
"	// reduction, phase 1\n" \
"	{\n" \
"		int i;\n" \
"		localSum = 0.0;\n" \
"		// for i = local index; i < cols; i += workGroupSize\n" \
"		for ( i = localIndex; i < cols; i = i + workGroupSize )\n" \
"		{\n" \
"			// sum += y[ i ] * alpha[ i ]\n" \
"			localSum += y[ i ] * alpha[ i ];\n" \
"		}\n" \
"		// scratch[ local index ] = sum\n" \
"		scratch[ localIndex ] = localSum;\n" \
"	}\n" \
"	barrier( CLK_LOCAL_MEM_FENCE );\n" \
"	// reduction, phase 2 (actually in parallel)\n" \
"	{\n" \
"		int offset;\n" \
"		// for offset = workGroupSize / 2; offset > 0; offset >>= 1\n" \
"		for ( offset = workGroupSize / 2; offset > 0; offset = offset >> 1 )\n" \
"		{\n" \
"			// if local index < offset\n" \
"			if ( localIndex < offset )\n" \
"			{\n" \
"				// scratch[ local index ] += scratch[ local index + offset ]\n" \
"				scratch[ localIndex ] += scratch[ localIndex + offset ];\n" \
"			}\n" \
"			barrier( CLK_LOCAL_MEM_FENCE );\n" \
"		}\n" \
"	}\n" \
"	// final assignment\n" \
"	{\n" \
"		// if 0 == local index\n" \
"		if ( 0 == localIndex )\n" \
"		{\n" \
"			// sum[ 0 ] = scratch[ 0 ]\n" \
"			sum[ 0 ] = scratch[ 0 ];\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
"\n" \
""
