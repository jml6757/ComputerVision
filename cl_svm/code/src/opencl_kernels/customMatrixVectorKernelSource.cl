"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"\n" \
"__kernel void custom_matrix_vector_kernel(\n" \
"											__global double * A,\n" \
"											__global double * x,\n" \
"											__global double * y,\n" \
"											//const int rows,\n" \
"											const int cols,\n" \
"											__local double * scratch,\n" \
"											//__global double * largerScratch,\n" \
"											const int localSize\n" \
"											//const int numberOfWorkGroups\n" \
"										)\n" \
"{\n" \
"\n" \
"	// variables\n" \
"	int rowNumber;\n" \
"	//int colNumber;\n" \
"	int localIndex;\n" \
"	//int localSize;\n" \
"	//int numberOfWorkGroups;\n" \
"	// function\n" \
"	// initialize\n" \
"	{ \n" \
"		// get row number \n" \
"		rowNumber = get_global_id( 0 ); \n" \
"		// get column number \n" \
"		//colNumber = get_global_id( 1 ); \n" \
"		localIndex = get_local_id( 1 ); \n" \
"		// get work group size \n" \
"		//localSize = get_local_size( 1 ); \n" \
"		//numberOfWorkGroups = get_global_size( 1 ) / localSize; \n" \
"	}\n" \
"	// phase 1 reduction\n" \
"	{\n" \
"		int startIndex = localIndex;\n" \
"		int aOffset = rowNumber * cols;\n" \
"		double sum = 0.0;\n" \
"		while ( startIndex < cols )\n" \
"		{\n" \
"			sum = sum + ( A[ aOffset + startIndex ] * x[ startIndex ] );\n" \
"			startIndex = startIndex + localSize;\n" \
"		}\n" \
"		scratch[ localIndex ] = sum;\n" \
"	}\n" \
"	// BARRIER \n" \
"	{ \n" \
"		barrier( CLK_LOCAL_MEM_FENCE ); \n" \
"	} \n" \
"	// REDUCE \n" \
"	{ \n" \
"		int offset = localSize / 2; \n" \
"		// for each power of two remaining \n" \
"		while ( offset > 0 ) \n" \
"		{ \n" \
"			// if we are in the lower half \n" \
"			if ( localIndex < offset ) \n" \
"			{ \n" \
"				scratch[ localIndex ] = scratch[ localIndex ] + scratch[ localIndex + offset ]; \n" \
"			} \n" \
"			barrier( CLK_LOCAL_MEM_FENCE ); \n" \
"			// divide by two \n" \
"			offset = offset >> 1; \n" \
"		} \n" \
"	}\n" \
"	// full assignment\n" \
"	{\n" \
"		if ( 0 == localIndex )\n" \
"		{\n" \
"			y[ rowNumber ] = scratch[ 0 ];\n" \
"		}\n" \
"	}\n" \
"\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
