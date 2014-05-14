#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void swap_objective_kernel(
										__global double * objectiveFunction,
										const int i,
										const int j
									 )
{
	// variables
	double tempI;
	double tempJ;
	
	// function
	// pretty straightforward: swap
	tempI = objectiveFunction[ i ];
	tempJ = objectiveFunction[ j ];
	// consider a barrier here
	// barrier( CLK_GLOBAL_MEM_FENCE );
	// I know it's a bit overkill, but it would allow for all elements to be swapped at once
	objectiveFunction[ i ] = tempJ;
	objectiveFunction[ j ] = tempI;
	// it is my understanding that all local variables are kept in up to 10 registers on
	// each core, so there's really no cost to the two variables, they should be as fast
	// as anything
	
	// clean up
	return;
}
