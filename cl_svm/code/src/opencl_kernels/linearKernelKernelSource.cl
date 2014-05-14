"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"#pragma OPENCL EXTENSION cl_amd_printf \n" \
"\n" \
"__kernel void linear_kernel_kernel(\n" \
"									__global double * x_data_i,\n" \
"									__global double * x_data_j,\n" \
"									__global double * outputData,\n" \
"									int goodDataSize,\n" \
"									__local double * scratchData\n" \
"								  )\n" \
"{\n" \
"	// variables\n" \
"	int localIndex;\n" \
"	int offset;\n" \
"	\n" \
"	// function\n" \
"	// get the index that we will work on\n" \
"	localIndex = get_local_id( 0 );\n" \
"	// perform our simple multiplication (this is a dot product, after all )\n" \
"	//if ( localIndex < goodDataSize )\n" \
"	{\n" \
"		//scratchData[ localIndex ] = x_data_i[ localIndex ] * x_data_j[ localIndex ];\n" \
"		outputData[ localIndex ] = x_data_i[ localIndex ] * x_data_j[ localIndex ];\n" \
"	}\n" \
"	/*else\n" \
"	{\n" \
"		// additive identity for padding\n" \
"		scratchData[ localIndex ] = 0.0;\n" \
"	}\n" \
"	// REDUCTION\n" \
"	barrier( CLK_LOCAL_MEM_FENCE );\n" \
"	// for each round\n" \
"	for ( offset = get_local_size(0) / 2; offset > 0; offset = offset >> 1 )\n" \
"	{\n" \
"		// if we are in this round\n" \
"		if ( localIndex < offset )\n" \
"		{\n" \
"			// add\n" \
"			scratchData[ localIndex ] = scratchData[ localIndex ] + scratchData[ localIndex + offset ];\n" \
"		}\n" \
"		// fence\n" \
"		barrier( CLK_LOCAL_MEM_FENCE );\n" \
"	}\n" \
"	// END OF REDUCTION\n" \
"	// assign output\n" \
"	if ( 0 == localIndex )\n" \
"	{\n" \
"		outputData[ 0 ] = scratchData[ 0 ];\n" \
"		//outputData[ 0 ] = 13.0;\n" \
"	}*/\n" \
"	\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
