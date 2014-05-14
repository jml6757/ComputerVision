"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" \
"\n" \
"__kernel void swap_objective_kernel(\n" \
"										__global double * objectiveFunction,\n" \
"										const int i,\n" \
"										const int j\n" \
"									 )\n" \
"{\n" \
"	// variables\n" \
"	double tempI;\n" \
"	double tempJ;\n" \
"	\n" \
"	// function\n" \
"	// pretty straightforward: swap\n" \
"	tempI = objectiveFunction[ i ];\n" \
"	tempJ = objectiveFunction[ j ];\n" \
"	// consider a barrier here\n" \
"	// barrier( CLK_GLOBAL_MEM_FENCE );\n" \
"	// I know it's a bit overkill, but it would allow for all elements to be swapped at once\n" \
"	objectiveFunction[ i ] = tempJ;\n" \
"	objectiveFunction[ j ] = tempI;\n" \
"	// it is my understanding that all local variables are kept in up to 10 registers on\n" \
"	// each core, so there's really no cost to the two variables, they should be as fast\n" \
"	// as anything\n" \
"	\n" \
"	// clean up\n" \
"	return;\n" \
"}\n" \
""
