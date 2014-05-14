"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"\n" \
"__kernel void swap_vector_block_kernel(\n" \
"										__global double * A,\n" \
"										const int rows,\n" \
"										const int cols,\n" \
"										const int i,\n" \
"										const int j\n" \
"									 )\n" \
"{\n" \
"	// variables\n" \
"	int globalIndex;\n" \
"	int matrixOffset1;\n" \
"	int matrixOffset2;\n" \
"	\n" \
"	// function\n" \
"	// figure out which index we're working on\n" \
"	{\n" \
"		// get our global id\n" \
"		globalIndex = get_global_id( 0 );\n" \
"		// calculate our offsets into the matrix\n" \
"		matrixOffset1 = i * cols + globalIndex;\n" \
"		matrixOffset2 = j * cols + globalIndex;\n" \
"	}\n" \
"	// swap two elements\n" \
"	{\n" \
"		double element1;\n" \
"		double element2;\n" \
"		// if our column is in the matrix\n" \
"		if ( globalIndex < cols )\n" \
"		{\n" \
"			// load the element we're responsible for from the first row\n" \
"			element1 = A[ matrixOffset1 ];\n" \
"			// load the element from the second row\n" \
"			element2 = A[ matrixOffset2 ];\n" \
"			// write them both back to the opposite rows\n" \
"			A[ matrixOffset1 ] = element2;\n" \
"			A[ matrixOffset2 ] = element1;\n" \
"		}\n" \
"	}\n" \
"	\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
