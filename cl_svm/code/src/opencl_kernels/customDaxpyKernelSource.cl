"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"\n" \
"__kernel void custom_daxpy_kernel(\n" \
"									__global double * x,\n" \
"									const double c\n" \
"								 )\n" \
"{\n" \
"	// variables\n" \
"	int localIndex;\n" \
"	\n" \
"	// function body\n" \
"	// get the index we're working on\n" \
"	localIndex = get_local_id( 0 );\n" \
"	// output[ index ] = input[ index ] + c\n" \
"	x[ localIndex ] = x[ localIndex ] + c;\n" \
"	\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
