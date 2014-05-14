//#include <cuda_runtime.h>
//#include "cublas_v2.h"

//#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include "clAmdBlas.h"
//#include "svm.h"

// Scalars
const float alpha = 1;
const float beta = 0;

// documentation
// http://docs.nvidia.com/cuda/cublas/

void ckm( struct svm_problem *prob, struct svm_problem *pecm, float *gamma  )
{
	//cublasStatus_t status;
	cl_int status;

	double g_val = *gamma;

	long int nfa;
	
	int len_tv;
	int ntv;
	int i_v;
	int i_el;
	int i_r, i_c;
	int trvei;

	double *tv_sq;
	double *v_f_g;

	float *tr_ar;
	float *tva, *vtm, *DP;
	// these all need to be cl_mem
	//float *g_tva = 0, *g_vtm = 0, *g_DotProd = 0;
	cl_mem g_tva;
	cl_mem g_vtm;
	cl_mem g_DotProd;

	//cudaError_t cudaStat;   
	//cublasHandle_t handle;
	cl_context svmContext;
	cl_command_queue svmQueue;
	cl_device_id firstDevice;
	cl_platform_id firstPlatform;
	cl_uint numberOfEntries;
	int errorCode;
	
	//status = cublasCreate(&handle);
	// get the device id
	if ( 0 != clGetPlatformIDs( 1, &firstPlatform, &numberOfEntries ) ||
		0 == numberOfEntries )
	{
		// TODO:
		// replace this placeholder
		// debugging
		fprintf( stderr, "Error getting platform IDs\n" );
		return;
	}
	if ( 0 != clGetDeviceIDs( firstPlatform, 
				CL_DEVICE_TYPE_GPU,
				1,
				&firstDevice,
				&numberOfEntries ) || 0 == numberOfEntries )
	{
		// TODO:
		// replace this placeholder
		// debugging
		fprintf( stderr, "Error getting device IDs\n" );
		return;
	}
	
	// create a compute context
	svmContext = clCreateContext( 0, 1, &firstDevice,
						NULL, NULL, &errorCode );
	if ( 0 != errorCode )
	{
		// TODO:
		// replace this placeholder
		// debugging
		fprintf( stderr, "Error creating context\n" );
		return;
	}
	svmQueue = clCreateCommandQueue( svmContext,
							  firstDevice,
							  0,
							  &errorCode );
	if ( 0 != errorCode )
	{
		// TODO:
		// replace this placeholder
		// debugging
		fprintf( stderr, "Error creating command queue\n" );
		return;
	}
	if ( CL_SUCCESS != clAmdBlasSetup() )
	{
		fprintf( stderr, "Error setting up cl blas\n" );
		return;
	}

	len_tv = prob-> x[0].dim;
	ntv   = prob-> l;

	nfa = len_tv * ntv; 

	tva = (float*) malloc ( len_tv * ntv* sizeof(float) );
	vtm = (float*) malloc ( len_tv * sizeof(float) );
	DP  = (float*) malloc ( ntv * sizeof(float) );

	tr_ar = (float*) malloc ( len_tv * ntv* sizeof(float) );

	tv_sq = (double*) malloc ( ntv * sizeof(double) );

	v_f_g  = (double*) malloc ( ntv * sizeof(double) );

	for ( i_r = 0; i_r < ntv ; i_r++ )
	{				 
		for ( i_c = 0; i_c < len_tv; i_c++ ) 
			tva[i_r * len_tv + i_c] = (float)prob-> x[i_r].values[i_c];
	}

	//cudaStat = cudaMalloc((void**)&g_tva, len_tv * ntv * sizeof(float));
	g_tva = clCreateBuffer( svmContext, CL_MEM_READ_WRITE, sizeof(float) * ntv * len_tv, NULL, &status );
	
	//if (cudaStat != cudaSuccess) {
	if ( 0 != status )
	{
		free( tva );
		free( vtm );
		free( DP  );

		free( v_f_g );
		free( tv_sq );

		//cudaFree( g_tva );
		clReleaseMemObject( g_tva );
		//cublasDestroy( handle );	
		clReleaseContext( svmContext );
		clReleaseCommandQueue( svmQueue );
	
		fprintf (stderr, "!!!! Device memory allocation error (A)\n");
		getchar();
		return;
    }

	//cudaStat = cudaMalloc((void**)&g_vtm, len_tv * sizeof(float));
	g_vtm = clCreateBuffer( svmContext, CL_MEM_READ_WRITE, sizeof(float) * len_tv, NULL, &status );

	//cudaStat = cudaMalloc((void**)&g_DotProd, ntv * sizeof(float));
	g_DotProd = clCreateBuffer( svmContext, CL_MEM_READ_WRITE, sizeof(float) * ntv, NULL, &status );

	for( i_r = 0; i_r < ntv; i_r++ )
		for( i_c = 0; i_c < len_tv; i_c++ )
			tr_ar[i_c * ntv + i_r] = tva[i_r * len_tv + i_c];

	// Copy cpu vector to gpu vector
	//status = cublasSetVector( len_tv * ntv, sizeof(float), tr_ar, 1, g_tva, 1 );
	status = clEnqueueWriteBuffer( svmQueue, g_tva, CL_TRUE, 0, len_tv * ntv * sizeof(float), tr_ar, 0, NULL, NULL );
    
	free( tr_ar );

	// this could probably stand to be on the gpu
	for( i_v = 0; i_v < ntv; i_v++ )
	{
		tv_sq[ i_v ] = 0;
		for( i_el = 0; i_el < len_tv; i_el++ )
			tv_sq[i_v] += pow( tva[i_v*len_tv + i_el], (float)2.0 );
	}



	for ( trvei = 0; trvei < ntv; trvei++ )
	{
		// copies len_tv elements of sizeof(float) from &tva[trvei * len_tv] (cpu space) to g_vtm (gpu space), strides of 1 and 1
		//status = cublasSetVector( len_tv, sizeof(float), &tva[trvei * len_tv], 1, g_vtm, 1 );
		status = clEnqueueWriteBuffer( svmQueue, g_vtm, CL_TRUE, 0, len_tv * sizeof(float), &tva[trvei * len_tv], 0, NULL, NULL );
		if ( 0 != status )
		{
			fprintf( stderr, "Error enqueuing write buffer\n" );
			return;
		}

		// this performs the operation y = alpha * op(A) * x + beta * y
		// output is stored in g_DotProd, probably the first phase of a dot product
		// CUBLAS_OP_N means no op, op(A) == A
		// g_DotProd = alpha * g_tva * g_vtm + beta * g_DotProd
		// beta is initialize to zero, alpha to 1
		// g_DotProd = g_tva * g_vtm
		// params: ( handle, op, rows, cols, alpha*, A*, lda (first dimension of output), x*, x_stride, beta*, y*, y_stride )
		//status = cublasSgemv( handle, CUBLAS_OP_N, ntv, len_tv, &alpha, g_tva, ntv , g_vtm, 1, &beta, g_DotProd, 1 );
		status = clAmdBlasSgemv( clAmdBlasRowMajor, clAmdBlasNoTrans, ntv, len_tv, alpha, g_tva, len_tv, g_vtm, 0, 1, beta, g_DotProd, 0, 1, 1, &svmQueue, 0, NULL, NULL );
		clFinish( svmQueue );
		if ( clAmdBlasSuccess != status )
		{
			fprintf( stderr, "Error doing stuff\n" );
			switch( status )
			{
				case clAmdBlasNotInitialized:
					fprintf( stderr, "Set up not called\n" );
					break;
				case clAmdBlasInvalidValue:
					fprintf( stderr, "Invalid value\n" );
					break;
				case clAmdBlasInvalidMemObject:
					fprintf( stderr, "Invalid mem object\n" );
					break;
				case clAmdBlasOutOfHostMemory:
					fprintf( stderr, "Out of host memory\n" );
					break;
				case clAmdBlasInvalidCommandQueue:
					fprintf( stderr, "Invalid command queue\n" );
					break;
				case clAmdBlasInvalidContext:
					fprintf( stderr, "Invalid context\n" );
					break;
				case clAmdBlasInvalidOperation:
					fprintf( stderr, "Invalid operation\n" );
					break;
				case clAmdBlasCompilerNotAvailable:
					fprintf( stderr, "Compiler not available\n" );
					break;
				case clAmdBlasBuildProgramFailure:
					fprintf( stderr, "Build program failure\n" );
					break;
				default: 
					fprintf( stderr, "Unrecognized error\n" );
					break;
			};
			return;
		}

		// copies ntv elements of sizeof(float) from g_DotProd (gpu space) to DP (cpu space), strides of 1 and 1
		//status = cublasGetVector( ntv, sizeof(float), g_DotProd, 1, DP, 1 );
		status = clEnqueueReadBuffer( svmQueue, g_DotProd, CL_TRUE, 0, ntv * sizeof(float), DP, 0, NULL, NULL );

		if ( 0 != status )
		{
			fprintf( stderr, "Error enqueuing read buffer\n" );
			return;
		}
		
		// I do have to wonder why this isn't on the GPU; possibly the exp() function
		for ( i_c = 0; i_c < ntv; i_c++ )
			v_f_g[i_c] = exp( -g_val * (tv_sq[trvei] + tv_sq[i_c]-((double)2.0)* (double)DP[i_c] ));
		

		pecm-> x[trvei].values[0] = trvei + 1;
		
		for ( i_c = 0; i_c < ntv; i_c++ )
			pecm-> x[trvei].values[i_c + 1] = v_f_g[i_c];				
		

	}

	free( tva );
	free( vtm );
	free( DP  );
	free( v_f_g );
	free( tv_sq );

	//cudaFree( g_tva );
	clReleaseMemObject( g_tva );
	//cudaFree( g_vtm );
	clReleaseMemObject( g_vtm );
	//cudaFree( g_DotProd );
	clReleaseMemObject( g_DotProd );
	clReleaseContext( svmContext );
	clReleaseCommandQueue( svmQueue );
	clAmdBlasTeardown();

	//cublasDestroy( handle );
}

void cal_km( struct svm_problem * p_km)
{
	float gamma = param.gamma;

	ckm(&prob, p_km, &gamma);
}