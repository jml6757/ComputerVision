"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"\n" \
"__kernel void dual_daxpy_kernel(\n" \
"									__global double * G,\n" \
"									//__global double * Q1,\n" \
"									__global float * Q1,\n" \
"									//__global double * Q2,\n" \
"									__global float * Q2,\n" \
"									const double alpha1,\n" \
"									//const float alpha1,\n" \
"									const double alpha2,\n" \
"									//const float alpha2,\n" \
"									const int activeSize\n" \
"								)\n" \
"{\n" \
"	// variables\n" \
"	int globalIndex = get_global_id( 0 );\n" \
"	\n" \
"	// function\n" \
"	// REALLY straightforward\n" \
"	//if ( (globalIndex = get_global_id( 0 )) < activeSize )\n" \
"	//while ( globalIndex < activeSize )\n" \
"	if ( globalIndex < activeSize )\n" \
"	{\n" \
"		G[ globalIndex ] += alpha1 * Q1[ globalIndex ] + alpha2 * Q2[ globalIndex ];\n" \
"		//globalIndex += groupSize;\n" \
"	}\n" \
"	\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
