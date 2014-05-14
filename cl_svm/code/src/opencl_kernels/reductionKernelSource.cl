"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"\n" \
"__kernel void reduction_kernel(\n" \
"								__global double * unreducedData,\n" \
"								const int cols,\n" \
"								__global double * reducedData\n" \
"							  )\n" \
"{\n" \
"\n" \
"	// variables\n" \
"	int globalIndex;\n" \
"	int offset;\n" \
"	double sum;\n" \
"\n" \
"	// function\n" \
"	// initialization\n" \
"	{\n" \
"		// get global index\n" \
"		globalIndex = get_global_id( 0 );\n" \
"		// get row based offset\n" \
"		offset = globalIndex * cols;\n" \
"	}\n" \
"	// add all the data together\n" \
"	{\n" \
"		sum = 0.0;\n" \
"		for ( int i = 0; i < cols; i++ )\n" \
"		{\n" \
"			sum += unreducedData[ offset + i ];\n" \
"		}\n" \
"	}\n" \
"	// assign output\n" \
"	{\n" \
"		reducedData[ globalIndex ] = sum;\n" \
"	}\n" \
"\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
