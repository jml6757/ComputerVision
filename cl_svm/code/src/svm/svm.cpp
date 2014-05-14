#include <math.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>

// OpenCL stuff
#include <CL/cl.h>
#include "clAmdBlas.h"
#include "gpu_cache.hpp"
#include "profiling.h"

#include <Windows.h>

#include "svm.h"
int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	
private:
	int l;
	long int size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	// switch actual vectors
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
			{
				// swap elements in other vectors to compensate
				swap(h->data[i],h->data[j]);
			}
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
	#ifdef CL_SVM

		// OpenCL related function
		virtual int update_objective_function( int activeSize, double deltaAlphaI, double deltaAlphaJ, int i, int j ) = 0;

		virtual int initialize_objective_function( int activeSize, const double * initialValue ) = 0;
		
		// TODO: MAKE FASTER
		/*inline*/virtual  int swap_objective_function( int i, int j ) = 0;
		
		// TODO: WRITE THIS ONCE THE OTHER FUNCTIONS HAVE BEEN VERIFIED
		//int reconstruct_objective_function( int activeSize, double * G_bar, 
		
		// THIS FUNCTION IS FOR TESTING
		virtual int check_objective_function( int activeSize, double * actualValues ) = 0;
		
		virtual int reconstruct_objective_function_part_one( int lowJ, int highJ, double * G_bar, double * p ) = 0;
		
		virtual int set_objective_function( int lowJ, int highJ, double * values ) = 0;
		
		virtual int retrieve_objective_function( int lowJ, int highJ, double * values ) = 0;
		
		virtual int select_working_set_i( const schar * y, char * alphaStatus, int activeSize, int &out_i, double &out_gmax ) = 0;
		
		virtual int select_working_set_j( const schar * y, char * alphaStatus, double gMax, int selectedI, double * QD, int activeSize, int & out_j, double & out_gMax2, double * outG ) = 0;
		
		virtual int swap_q_indices( int i, int j ) = 0;
		
	#endif
	
};

// try some logging
void * pfn_notify( const char * errInfo, const void * private_info, size_t cb, void * user_data )
{
	// variables
	
	// function body
	// debugging
	fprintf( stderr, "Error in kernel context: %s\n", errInfo );
	
	// clean up
	return NULL;
}

// profiling variables
LONGLONG writingVectorsTime = 0;
LONGLONG outputSetupTime = 0;
LONGLONG blasExecutionTime = 0;
LONGLONG outputReadingTime = 0;

class Kernel: public QMatrix {

//#define	GPU_CACHE_SIZE	10

public:

	// variables
	int wideKernelInUse;
	int numberOfVectors;
	
	// methods

#ifdef _DENSE_REP
	Kernel(int l, svm_node * x, const svm_parameter& param);
#else
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
#endif
	virtual ~Kernel();

	#ifdef CL_SVM
		double wide_k_function( const svm_node * x, const svm_node * y,
										const svm_parameter & param, double * svmCoefficients );
	#endif
	
	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
	
		// profiling
		profileDecls;
		Kernel * copy;
		copy = (Kernel*)this;
	
		// TODO: Introduce GPU swap
		//gpuCache.InvalidateCache();
		((GPUCache)gpuCache).SwapIndices( i, j );
		copy->swap_vector_block_indices( i, j );
		//((GPUCache)gpuQCache).InvalidateCache();
		((GPUCache)cpuQCache).InvalidateCache();
		//gpuQCache.SwapIndices( i, j );
		//startTimer();
		/*if ( 0 != swap_q_indices( i, j ) )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR SWAPPING Q INDICES\n" );
			exit( -1 );
		}*/
		//stopTimer();
		//swappingTime += calculateTime();
		
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
	
	#ifdef CL_SVM
		// TODO: Tune this according to device info
		#define	IDEAL_WORK_GROUP_SIZE	128
		#define	CL_PRECISION_ERROR	0.001
	
	// FOR PROFILING: TEMPORARY
	LONGLONG otherFunctionTime;// = 0;
	LONGLONG gettingQTime;// = 0;
	LONGLONG gettingGTime;// = 0;
	LONGLONG daxpyTime;// = 0;
	LONGLONG swappingTime;// = 0;
	LONGLONG swappingCount;// = 0;
	LONGLONG qRetrievingTime;// = 0;
	LONGLONG swappingOtherFunctionTime;// = 0;
	LONGLONG qCacheSwappingTime;// = 0;
	LONGLONG endQueueTime;// = 0;
	LONGLONG kernelEnqueuingTime;// = 0;
	LONGLONG argSettingTime;// = 0;
	
		// OpenCL related function
		int update_objective_function( int activeSize, double deltaAlphaI, double deltaAlphaJ, int i, int j )
		{
			
			// variables
			cl_mem q_i;
			cl_mem q_j;
			cl_mem g;
			cl_int errorCode;
			cl_int errorCodeInt;
			profileDecls;
			float * qVector1;
			//double * qVector1;
			float * qVector2;
			//double * qVector2;
			int workDimension;
			int retried;
			size_t globalWorkSize[3];
			size_t localWorkSize[3];
			
			// function body
			
			// just for safety
			startTimer();
			{
				clFinish( kernelCommandQueue );
				if ( CL_SUCCESS != clFinish( kernelCpuCommandQueue ) )
				{
					fprintf( stderr, "ERROR FINISHING CPU QUEUE\n" );
					return -1;
				}
			}
			stopTimer();
			otherFunctionTime += calculateTime();
			// fetch preallocated memory
			/*{
				startTimer();
				// if we can't get Q values
				if ( 0 != gpuQCache.CheckCache( i, &q_i ) )
				{
					// return error
					//fprintf( stderr, "WARNING: Failed to find cached kernel matrix when updating objective function. Highly unexpected.\n" );
					//return -1;
					(this->*wide_kernel_function)( i, 0, numberOfVectors-1, NULL );
					if ( 0 != gpuQCache.CheckCache( i, &q_i ) )
					{
						// THIS IS VERY SERIOUS
						fprintf( stderr, "ERROR: VECTOR IS STILL NOT IN CACHE, PANICKING: %i\n", i );
						return -1;
					}
				}
				if ( 0 != gpuQCache.CheckCache( j, &q_j ) )
				{
					// return error
					//fprintf( stderr, "WARNING: Failed to find cached kernel matrix when updating objective function. Highly unexpected. %i\n", j );
					//return -1;
					//get_Q( j, activeSize );
					(this->*wide_kernel_function)( j, 0, numberOfVectors-1, NULL );
					if ( 0 != gpuQCache.CheckCache( j, &q_j ) )
					{
						// THIS IS VERY SERIOUS
						fprintf( stderr, "ERROR: VECTOR IS STILL NOT IN CACHE, PANICKING: %i\n", j );
						return -1;
					}
				}
				stopTimer();
				gettingQTime += calculateTime();
				startTimer();
				// if we can't get G data
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// return error
					fprintf( stderr, "WARNING: Failed to find reserved space for objective function. Highly unexpected.\n" );
					return -1;
				}
				stopTimer();
				gettingGTime += calculateTime();
			}*/
			retried = 0;
update_objective_function_memory_retry_label:
			startTimer();
			// calculate the q vectors newly
			/*{
				qVector1 = (double*) malloc( sizeof( double ) * numberOfVectors );
				qVector2 = (double*) malloc( sizeof( double ) * numberOfVectors );
				(this->*wide_kernel_function)( i, 0, numberOfVectors-1, qVector1 );
				(this->*wide_kernel_function)( j, 0, numberOfVectors-1, qVector2 );
			}
			*/
			// write them out to the CPU
			{
				// create buffers for the two q vectors
				if ( 0 != cpuQCache.CheckCache( i, &q_i ) )
				{
					qVector1 = get_Q( i, activeSize );
					q_i = clCreateBuffer( kernelCpuContext, CL_MEM_READ_ONLY, sizeof(float) * numberOfVectors,//sizeof(float) * activeSize,
										  NULL, &errorCodeInt );
					if ( CL_SUCCESS != errorCodeInt )
					{
						fprintf( stderr, "ERROR CREATING BUFFER FOR QI IN OBJECTIVE FUNCTION UPDATE\n" );
						switch( errorCodeInt )
						{
							case CL_INVALID_CONTEXT:
								fprintf( stderr, "CL_INVALID_CONTEXT\n" );
								break;
							case CL_INVALID_VALUE:
								fprintf( stderr, "CL_INVALID_VALUE\n" );
								break;
							case CL_INVALID_BUFFER_SIZE:
								fprintf( stderr, "CL_INVALID_BUFFER_SIZE\n" );
								break;
							case CL_INVALID_HOST_PTR:
								fprintf( stderr, "CL_INVALID_HOST_PTR\n" );
								break;
							case CL_MEM_OBJECT_ALLOCATION_FAILURE:
								fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
								break;
							case CL_OUT_OF_HOST_MEMORY:
								fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
								stopTimer();
								if ( retried )
								{
									fprintf( stderr, "GIVING UP\n" );
									return -1;
								}
								fprintf( stderr, "CLEARING MEMORY, THEN TRYING AGAIN\n" );
								retried = 1;
								cpuQCache.InvalidateCache();
								gpuCache.InvalidateCache();
								goto update_objective_function_memory_retry_label;
								break;
							default:
								fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCodeInt );
								break;
						};
						return -1;
					}
					errorCode = clEnqueueWriteBuffer( kernelCpuCommandQueue, q_i, CL_FALSE, 0, 
														sizeof(float) * activeSize, qVector1, 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						fprintf( stderr, "ERROR WRITING QI VECTOR TO CPU QUEUE\n" );
						return -1;
					}
					if ( 0 != cpuQCache.CacheData( i, q_i ) )
					{
						// THIS IS VERY SERIOUS
						fprintf( stderr, "ERROR: CACHING I VECTOR PANICKING: %i\n", i );
						return -1;
					}
				}
				if ( 0 != cpuQCache.CheckCache( j, &q_j ) )
				{
					qVector2 = get_Q( j, activeSize );
					q_j = clCreateBuffer( kernelCpuContext, CL_MEM_READ_ONLY, sizeof(float) * numberOfVectors,//sizeof(float) * activeSize,
										  NULL, &errorCodeInt );
					if ( CL_SUCCESS != errorCodeInt )
					{
						fprintf( stderr, "ERROR CREATING BUFFER FOR QJ IN OBJECTIVE FUNCTION UDPATE\n" );
						switch( errorCodeInt )
						{
							case CL_INVALID_CONTEXT:
								fprintf( stderr, "CL_INVALID_CONTEXT\n" );
								break;
							case CL_INVALID_VALUE:
								fprintf( stderr, "CL_INVALID_VALUE\n" );
								break;
							case CL_INVALID_BUFFER_SIZE:
								fprintf( stderr, "CL_INVALID_BUFFER_SIZE\n" );
								break;
							case CL_INVALID_HOST_PTR:
								fprintf( stderr, "CL_INVALID_HOST_PTR\n" );
								break;
							case CL_MEM_OBJECT_ALLOCATION_FAILURE:
								fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
								break;
							case CL_OUT_OF_HOST_MEMORY:
								fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
								stopTimer();
								if ( retried )
								{
									fprintf( stderr, "GIVING UP\n" );
									return -1;
								}
								retried = 1;
								fprintf( stderr, "CLEARING MEMORY, THEN TRYING AGAIN\n" );
								cpuQCache.InvalidateCache();
								gpuCache.InvalidateCache();
								goto update_objective_function_memory_retry_label;
								break;
							default:
								fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCodeInt );
								break;
						};
						return -1;
					}
					// write to them
					
					errorCode = clEnqueueWriteBuffer( kernelCpuCommandQueue, q_j, CL_FALSE, 0, 
														sizeof(float) * activeSize/*sizeof(float) * activeSize*/, qVector2, 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						fprintf( stderr, "ERROR WRITING QJ VECTOR TO CPU QUEUE\n" );
						return -1;
					}
					if ( 0 != cpuQCache.CacheData( j, q_j ) )
					{
						// THIS IS VERY SERIOUS
						fprintf( stderr, "ERROR: CACHING J VECTOR PANICKING: %i\n", j );
						return -1;
					}
				}
			}
			// wait for the rights to finish
			{
				clFinish( kernelCpuCommandQueue );
			}
			stopTimer();
			gettingQTime += calculateTime();
			// get G
			{
				startTimer();
				// if we can't get G data
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// return error
					fprintf( stderr, "WARNING: Failed to find reserved space for objective function. Highly unexpected.\n" );
					return -1;
				}
				stopTimer();
				gettingGTime += calculateTime();
			}
			startTimer();
			// execute update using Daxpy
			// This takes the most time in this function....that's so upsetting
			/*{
				// G = deltaAlphaI * Qi + G
				// shortcut
				//if ( 0.0 != deltaAlphaI )
				{
					errorCode = clAmdBlasDaxpy( activeSize, deltaAlphaI, q_i, 0, 1, g, 0, 1, 1, &kernelCommandQueue, 0, NULL, NULL );
					if ( clAmdBlasSuccess != errorCode )
					{
						// TODO: Deal with this error
						fprintf( stderr, "ERROR PERFORMING DAXPY FUNCTION WHILE UPDATING OBJECTIVE FUNCTION\n" );
						return -1;
					}
					// I wonder if we get the same result without this one
					//clFinish( kernelCommandQueue );
				}
				// G = deltaAlphaJ * Qj + G
				// shortcut
				//if ( 0.0 != deltaAlphaJ )
				{
					errorCode = clAmdBlasDaxpy( activeSize, deltaAlphaJ, q_j, 0, 1, g, 0, 1, 1, &kernelCommandQueue, 0, NULL, NULL );
					if ( clAmdBlasSuccess != errorCode )
					{
						// TODO: Deal with this error
						fprintf( stderr, "ERROR PERFORMING DAXPY FUNCTION WHILE UPDATING OBJECTIVE FUNCTION\n" );
						return -1;
					}
					clFinish( kernelCommandQueue );
				}
			}*/
			// set up call to custom kernel
			{
				// set up work dimensions
				int remainder = activeSize % IDEAL_WORK_GROUP_SIZE;
				float floatA1 = deltaAlphaI;
				float floatA2 = deltaAlphaJ;
				workDimension = 1;
				globalWorkSize[0] = activeSize + IDEAL_WORK_GROUP_SIZE - remainder;
				//globalWorkSize[0] = IDEAL_WORK_GROUP_SIZE;
				// this may fail or have to be adjusted
				localWorkSize[0] = IDEAL_WORK_GROUP_SIZE;
				// set up arguments
				// call will be of the form: kernel( G, Q1, Q2, alpha1, alpha2, activeSize )
				errorCode = clSetKernelArg( dualDaxpyKernel, 0, sizeof(cl_mem), &g );
				errorCode |= clSetKernelArg( dualDaxpyKernel, 1, sizeof(cl_mem), &q_i );
				errorCode |= clSetKernelArg( dualDaxpyKernel, 2, sizeof(cl_mem), &q_j );
				errorCode |= clSetKernelArg( dualDaxpyKernel, 3, sizeof(double), &deltaAlphaI );
				errorCode |= clSetKernelArg( dualDaxpyKernel, 4, sizeof(double), &deltaAlphaJ );
				errorCode |= clSetKernelArg( dualDaxpyKernel, 5, sizeof(int), &activeSize );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR SETTING KERNEL ARGS FOR DUAL DAXPY\n" );
					return -1;
				}
			}
			// call the custom kernel
			{
				//errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, dualDaxpyKernel, workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
				errorCode = clEnqueueNDRangeKernel( kernelCpuCommandQueue, dualDaxpyKernel, workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR RUNNING ND RANGE KERNEL\n" );
					switch( errorCode )
				{
					case CL_INVALID_PROGRAM_EXECUTABLE:
						fprintf( stderr, "CL_INVALID_PROGRAM_EXECUTABLE\n" );
						break;
					case CL_INVALID_COMMAND_QUEUE :
						fprintf( stderr, "CL_INVALID_COMMAND_QUEUE \n" );
						break;
					case CL_INVALID_KERNEL:
						fprintf( stderr, "CL_INVALID_KERNEL\n" );
						break;
					case CL_INVALID_CONTEXT:
						fprintf( stderr, "CL_INVALID_CONTEXT\n" );
						break;
					case CL_INVALID_KERNEL_ARGS:
						fprintf( stderr, "CL_INVALID_KERNEL_ARGS\n" );
						break;
					case CL_INVALID_WORK_DIMENSION:
						fprintf( stderr, "CL_INVALID_WORK_DIMENSION\n" );
						break;
					case CL_INVALID_WORK_GROUP_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_GROUP_SIZE\n" );
						break;
					case CL_INVALID_WORK_ITEM_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_ITEM_SIZE\n" );
						break;
					case CL_INVALID_GLOBAL_OFFSET:
						fprintf( stderr, "CL_INVALID_GLOBAL_OFFSET\n" );
						break;
					case CL_OUT_OF_RESOURCES:
						fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
						break;
					case CL_MEM_OBJECT_ALLOCATION_FAILURE:
						fprintf( stderr, "RAN OUT OF MEMORY\n" );
						if ( retried )
						{
							fprintf( stderr, "GIVING UP\n" );
							return -1;
						}
						fprintf( stderr, "CLEARING MEMORY, THEN TRYING AGAIN\n" );
						retried = 1;
						cpuQCache.InvalidateCache();
						gpuCache.InvalidateCache();
						goto update_objective_function_memory_retry_label;
						break;
					case CL_INVALID_EVENT_WAIT_LIST:
						fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
						break;
					case CL_OUT_OF_HOST_MEMORY:
						fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
						break;
					default:
						fprintf( stderr, "OTHER FAILURE: %i\n", errorCode );
						break;
				};
					return -1;
				}
				//clFinish( kernelCommandQueue );
				clFinish( kernelCpuCommandQueue );
			}
			stopTimer();
			daxpyTime += calculateTime();
			
			
			// clean up
			{
				//clReleaseMemObject( q_i );
				//clReleaseMemObject( q_j );
				//free( qVector1 );
				//free( qVector2 );
			}
			return 0;
		}

		int initialize_objective_function( int activeSize, const double * initialValue )
		{
		
			// variables
			cl_mem g;
			cl_int errorCode;
			
			// function body
			// for safety, clear the queue
			{
				clFinish( kernelCommandQueue );
				clFinish( kernelCpuCommandQueue );
			}
			// allocate space for G
			{
				int remainder = activeSize % IDEAL_WORK_GROUP_SIZE;
				//g = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * (activeSize + IDEAL_WORK_GROUP_SIZE - remainder), NULL, &errorCode );
				g = clCreateBuffer( kernelCpuContext, CL_MEM_READ_WRITE, sizeof(double) * (activeSize + IDEAL_WORK_GROUP_SIZE - remainder), NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING SPACE FOR G IN INITIALIZATION\n" );
					return -1;
				}
			}
			// write that data out to the GPU
			{
				// may not have to be a blocking call
				// TODO: Make sure this desynchronization doesn't mess us up later on
				//errorCode = clEnqueueWriteBuffer( kernelCommandQueue, g, CL_FALSE, 0, sizeof(double) * activeSize, initialValue, 0, NULL, NULL );
				errorCode = clEnqueueWriteBuffer( kernelCpuCommandQueue, g, CL_FALSE, 0, sizeof(double) * activeSize, initialValue, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING DATA WHILE INITIALIZING OBJECTIVE FUNCTION\n" );
					return -1;
				}
			}
			// cache it
			{
				if ( 0 != gpuCache.SaveGSpace( g ) )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CACHING OBJECTIVE FUNCTION AT INITIALIZATION\n" );
					return -1;
				}
			}
			
			// clean up
			return 0;
		}

		/*inline*/int swap_objective_function( int i, int j )
		{
			// variables
			cl_int errorCode;
			cl_mem g;
			cl_uint workDimension;
			size_t globalWorkSize[3];
			size_t localWorkSize[3];
			
			// function body
			// for safety, clear the queue
			{
				clFinish( kernelCommandQueue );
				clFinish( kernelCpuCommandQueue );
			}
			// conjure up preallocated memory
			{
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR: OBJECTIVE FUNCTION SPACE DOES NOT EXIST BEFORE SWAP. HIGHLY UNEXPECTED\n" );
					return -1;
				}
			}
			// set up the call to the kernel
			{
				// kernel should take pointer to G
				errorCode = clSetKernelArg( swapObjectiveFunctionKernel, 0, sizeof(cl_mem), &g );
				// and indices
				errorCode |= clSetKernelArg( swapObjectiveFunctionKernel, 1, sizeof(int), &i );
				errorCode |= clSetKernelArg( swapObjectiveFunctionKernel, 2, sizeof(int), &j );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR: FAILED TO SET OBJECTIVE FUNCTION SWAPPING KERNEL ARGUMENTS\n" );
					return -1;
				}
				// set up work dimensions
				workDimension = 1;
				globalWorkSize[0] = 1;
				localWorkSize[0] = 1;
			}
			// call the kernel
			{
				//errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, swapObjectiveFunctionKernel, workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
				errorCode = clEnqueueNDRangeKernel( kernelCpuCommandQueue, swapObjectiveFunctionKernel, workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR RUNNING OBJECTIVE FUNCTION SWAP KERNEL\n" );
				}
			}
			
			// clean up
			return 0;
		}
		
		// THIS FUNCTION IS FOR TESTING
		int check_objective_function( int activeSize, double * actualValues )
		{
			// variables
			cl_int errorCode;
			cl_mem g;
			uint32_t i;
			double * gpuData;
			
			// function body
			// allocate CPU space for objective function
			{
				gpuData = (double*) malloc( sizeof(double) * activeSize );
				if ( NULL == gpuData )
				{
					fprintf( stderr, "ERROR OCCURED ALLOCATING SPACE TO CHECK OBJECTIVE FUNCTION\n" );
					return -1;
				}
			}
			// for safety, empty the queue
			{
				clFinish( kernelCommandQueue );
				clFinish( kernelCpuCommandQueue );
			}
			// try to get a hold of the space for G
			{
				// if we can't get G data
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// return error
					fprintf( stderr, "WARNING: Failed to find reserved space for objective function while checking. Highly unexpected.\n" );
					return -1;
				}
			}
			// read objective function off of GPU
			{
				//errorCode = clEnqueueReadBuffer( kernelCommandQueue, g, CL_TRUE, 0, sizeof(double) * activeSize, gpuData, 0, NULL, NULL );
				errorCode = clEnqueueReadBuffer( kernelCpuCommandQueue, g, CL_TRUE, 0, sizeof(double) * activeSize, gpuData, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR READING G DATA WHILE CHECKING OBJECTIVE FUNCTION\n" );
					return -1;
				}
			}
			// check it against supplied values
			{
				for ( i = 0; i < activeSize; i++ )
				{
					//if ( (float)gpuData[ i ] != (float)actualValues[ i ] )
					if ( CL_PRECISION_ERROR < std::abs( gpuData[ i ] - actualValues[ i ] ) )
					{
						fprintf( stderr, "AT INDEX %i, GPU AND CPU DATA FOR OBJECTIVE FUNCTION DO NOT MATCH: %lf != %lf\n", i, gpuData[ i ], actualValues[ i ] );
						return -1;
					}
				}
			}
			
			// clean up
			{
				free( gpuData );
			}
			return 0;
		}
		
		int reconstruct_objective_function_part_one( int lowJ, int highJ, double * G_bar, double * p )
		{
			// variables
			cl_int errorCode;
			cl_mem g;
			cl_mem gBarGpu;
			
			// function
			// to be safe, empty the queue
			{
				clFinish( kernelCommandQueue );
			}
			// get cached G data
			{
				// if we can't get G data
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// return error
					fprintf( stderr, "WARNING: Failed to find reserved space for objective function. Highly unexpected.\n" );
					return -1;
				}
			}
			// set up input data structures
			{
				gBarGpu = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * (highJ - lowJ), NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error in earnest
					fprintf( stderr, "ERROR: Failed to create G_bar buffer during reconstruction part 1\n" );
					return -1;
				}
				/*pGpu = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * (highJ - lowJ), NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error in earnest
					fprintf( stderr, "ERROR: Failed to create p buffer during reconstruction part 1\n" );
					return -1;
				}*/
			}
			// write p to g's space
			{
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, g, CL_TRUE, sizeof(double) * lowJ, sizeof(double) * (highJ - lowJ), p, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error in earnest
					fprintf( stderr, "ERROR WRITING P DATA TO GPU\n" );
					return -1;
				}
			}
			// write gBar to gBar's space
			{
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, gBarGpu, CL_TRUE, 0, sizeof(double) * (highJ - lowJ), G_bar, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error in earnest
					fprintf( stderr, "ERROR WRITING G_BAR DATA TO GPU\n" );
					return -1;
				}
			}
			// call DAXPY
			{
				errorCode = clAmdBlasDaxpy( (highJ - lowJ), 1.0, gBarGpu, 0, 1, g, lowJ, 1, 1, &kernelCommandQueue, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR RECONSTRUCTING G\n" );
					return -1;
				}
				clFinish( kernelCommandQueue );
			}
			
			// clean up
			{
				clReleaseMemObject( gBarGpu );
			}
			return 0;
		}
		
		int set_objective_function( int lowJ, int highJ, double * values )
		{
			// variables
			cl_int errorCode;
			cl_mem g;
			
			// function body
			// to be safe, empty the queue
			{
				clFinish( kernelCommandQueue );
				clFinish( kernelCpuCommandQueue );
			}
			// get cached g
			{
				// if we can't get G data
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// return error
					fprintf( stderr, "WARNING: Failed to find reserved space for objective function. Highly unexpected.\n" );
					return -1;
				}
			}
			// write to it
			{
				//errorCode = clEnqueueWriteBuffer( kernelCommandQueue, g, CL_TRUE, sizeof(double) * lowJ, sizeof(double) * (highJ - lowJ), values, 0, NULL, NULL );
				errorCode = clEnqueueWriteBuffer( kernelCpuCommandQueue, g, CL_TRUE, sizeof(double) * lowJ, sizeof(double) * (highJ - lowJ), values, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING TO SET OBJECTIVE FUNCTION\n" );
					return -1;
				}
			}
			
			// clean up
			return 0;
		}
		
		int retrieve_objective_function( int lowJ, int highJ, double * values )
		{
			// variables
			cl_int errorCode;
			cl_mem g;
			
			// function body
			// to be safe, empty the queue
			{
				clFinish( kernelCommandQueue );
				clFinish( kernelCpuCommandQueue );
			}
			// get cached g
			{
				// if we can't get G data
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// return error
					fprintf( stderr, "WARNING: Failed to find reserved space for objective function. Highly unexpected.\n" );
					return -1;
				}
			}
			// read it into the pointer provided
			{
				//errorCode = clEnqueueReadBuffer( kernelCommandQueue, g, CL_TRUE, sizeof(double) * lowJ, sizeof(double) * (highJ - lowJ), values, 0, NULL, NULL );
				errorCode = clEnqueueReadBuffer( kernelCpuCommandQueue, g, CL_TRUE, sizeof(double) * lowJ, sizeof(double) * (highJ - lowJ), values, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR READING G BACK TO CPU\n" );
					return -1;
				}
			}
			
			// clean up
			return 0;
		}
		
		int select_working_set_i( const schar * y, char * alphaStatus, int activeSize, int & out_i, double & out_gmax )
		{
			// variables
			cl_int errorCode;
			cl_mem g;
			cl_mem yGpu;
			cl_mem alphaStatusGpu;
			cl_mem valueBuffer;
			cl_mem indexBuffer;
			size_t globalWorkSize[3];
			size_t localWorkSize[3];
			uint32_t i;
			double * cpuValues;
			int * cpuIndices;
			int workDimension;
			int numberOfWorkGroups;

			
			// function body
			// for safety, clear queue
			{
				clFinish( kernelCommandQueue );
			}
			// get cached g buffer
			{
				// if we can't get G data
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// return error
					fprintf( stderr, "WARNING: Failed to find reserved space for objective function. Highly unexpected.\n" );
					return -1;
				}
			}
			// create buffers for y and alpha status
			{
				// calculate the work group sizes
				int remainder = activeSize % IDEAL_WORK_GROUP_SIZE;
				globalWorkSize[0] = activeSize + IDEAL_WORK_GROUP_SIZE - remainder;
				numberOfWorkGroups = globalWorkSize[0] / IDEAL_WORK_GROUP_SIZE;
				// create y buffer
				// TODO: Allocate this once
				//if ( -1 == gpuCache.GetYSpace( &yGpu ) )
				{
					yGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(schar) * globalWorkSize[0], NULL, &errorCode );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this
						fprintf( stderr, "ERROR CREATING SPACE FOR Y DATA\n" );
						return -1;
					}
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, yGpu, CL_FALSE, 0, sizeof(schar) * activeSize, y, 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this error
						fprintf( stderr, "ERROR WRITING Y VALUE TO GPU\n" );
						return -1;
					}
					//gpuCache.SaveYSpace( yGpu );
				}
				// create alpha status buffer
				alphaStatusGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(char) * globalWorkSize[0], NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING SPACE FOR ALPHA STATUS\n" );
					return -1;
				}
			}
			// create output buffers for kernel
			{
				// two buffers, one for indices, one for values
				valueBuffer = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * numberOfWorkGroups, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING SPACE FOR VALUE BUFFER\n" );
					return -1;
				}
				indexBuffer = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(int) * numberOfWorkGroups, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING SPACE FOR INDEX BUFFER\n" );
					return -1;
				}
			}
			// write y and alpha status vectors
			{
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, alphaStatusGpu, CL_FALSE, 0, sizeof(char) * activeSize, alphaStatus, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING ALPHA STATUS TO GPU\n" );
					return -1;
				}
			}
			// set up call to kernel
			{
				// set up local work size, global size is already set up
				workDimension = 1;
				localWorkSize[0] = IDEAL_WORK_GROUP_SIZE;
				// call is of the form: kernel( g, y, alpha_status, activeSize, indexBuffer, valueBuffer )
				errorCode = clSetKernelArg( findCandidateIValuesKernel, 0, sizeof(cl_mem), &g );
				errorCode |= clSetKernelArg( findCandidateIValuesKernel, 1, sizeof(cl_mem), &yGpu );
				errorCode |= clSetKernelArg( findCandidateIValuesKernel, 2, sizeof(cl_mem), &alphaStatusGpu );
				errorCode |= clSetKernelArg( findCandidateIValuesKernel, 3, sizeof(int), &activeSize );
				errorCode |= clSetKernelArg( findCandidateIValuesKernel, 4, sizeof(cl_mem), &indexBuffer );
				errorCode |= clSetKernelArg( findCandidateIValuesKernel, 5, sizeof(cl_mem), &valueBuffer );
				errorCode |= clSetKernelArg( findCandidateIValuesKernel, 6, sizeof(int) * IDEAL_WORK_GROUP_SIZE, NULL );
				errorCode |= clSetKernelArg( findCandidateIValuesKernel, 7, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR SETTING KERNEL ARGS FOR FINDING I CANDIDATES\n" );
					return -1;
				}
			}
			// call kernel
			{
				errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, findCandidateIValuesKernel, workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR RUNNING KERNEL TO FIND I CANDIDATES\n" );
					switch( errorCode )
					{
						case CL_INVALID_PROGRAM_EXECUTABLE:
							fprintf( stderr, "CL_INVALID_PROGRAM_EXECUTABLE\n" );
							break;
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "CL_INVALID_COMMAND_QUEUE\n" );
							break;
						case CL_INVALID_KERNEL:
							fprintf( stderr, "CL_INVALID_KERNEL\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_KERNEL_ARGS:
							fprintf( stderr, "CL_INVALID_KERNEL_ARGS\n" );
							break;
						case CL_INVALID_WORK_DIMENSION:
							fprintf( stderr, "CL_INVALID_WORK_DIMENSION\n" );
							break;
						case CL_INVALID_WORK_GROUP_SIZE:
							fprintf( stderr, "CL_INVALID_WORK_GROUP_SIZE\n" );
							break;
						case CL_INVALID_WORK_ITEM_SIZE :
							fprintf( stderr, "CL_INVALID_WORK_ITEM_SIZE \n" );
							break;
						case CL_INVALID_GLOBAL_OFFSET:
							fprintf( stderr, "CL_INVALID_GLOBAL_OFFSET\n" );
							break;
						case CL_OUT_OF_RESOURCES:
							fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
					};
					return -1;
				}
				errorCode = clFinish( kernelCommandQueue );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR FINISHING KERNEL\n" );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
					};
					return -1;
				}
			}
			// read back output from kernel
			{
				// make space on the CPU for them
				cpuValues = (double*) malloc( sizeof(double) * numberOfWorkGroups );
				if ( NULL == cpuValues )
				{
					// TODO: Deal with this for real
					fprintf( stderr, "ERROR ALLOCATING CPU SPACE FOR VALUE BUFFER\n" );
					return -1;
				}
				cpuIndices = (int*) malloc( sizeof(int) * numberOfWorkGroups );
				if ( NULL == cpuIndices )
				{
					// TODO: Deal with this for real
					fprintf( stderr, "ERROR ALLOCATING CPU SPACE FOR INDEX BUFFER\n" );
					return -1;
				}
				// read them
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, valueBuffer, CL_FALSE, 0, sizeof(double) * numberOfWorkGroups, cpuValues, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR READING VALUE BUFFER\n" );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, indexBuffer, CL_FALSE, 0, sizeof(int) * numberOfWorkGroups, cpuIndices, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR READING INDEX BUFFER\n" );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
			}
			// wait for the reads to be done
			{
				clFinish( kernelCommandQueue );
			}
			// reduce it on the CPU
			{
				int maxIndex = -1;
				double maxValue = -INF;
				for ( i = 0; i < numberOfWorkGroups; i++ )
				{
					if ( cpuValues[ i ] > maxValue )
					{
						maxIndex = cpuIndices[ i ];
						maxValue = cpuValues[ i ];
					}
				}
				out_i = maxIndex;
				out_gmax = maxValue;
			}
			
			// clean up
			{
				clReleaseMemObject( yGpu );
				clReleaseMemObject( alphaStatusGpu );
				clReleaseMemObject( valueBuffer );
				clReleaseMemObject( indexBuffer );
				free( cpuValues );
				free( cpuIndices );
			}
			return 0;
		}
		
		int select_working_set_j( const schar * y, char * alphaStatus, double gMax, int selectedI, double * QD, int activeSize, int & out_j, double & out_gMax2, double * outG )
		{
			// variables
			cl_int errorCode;
			cl_mem alphaStatusGpu;
			cl_mem g;
			cl_mem qdGpu;
			cl_mem q_i;
			cl_mem yGpu;
			cl_mem gMaxGpu;
			cl_mem valueBuffer;
			cl_mem indexBuffer;
			size_t globalWorkSize[3];
			size_t localWorkSize[3];
			uint32_t i;
			double * gmax2Candidates;
			double * cpuValues;
			int * cpuIndices;
			int numberOfWorkGroups;
			int workDimension;
			
			
			// function body
			// just to be safe, clear the queue
			{
				clFinish( kernelCommandQueue );
			}
			// get cached g
			{
				// if we can't get G data
				if ( 0 != gpuCache.GetGSpace( &g ) )
				{
					// return error
					fprintf( stderr, "WARNING: Failed to find reserved space for objective function. Highly unexpected.\n" );
					return -1;
				}
			}
			// get cached Q_i
			{
				// if we can't get Q values
				if ( 0 != gpuQCache.CheckCache( selectedI, &q_i ) )
				{
					// return error
					//fprintf( stderr, "WARNING: Failed to find cached kernel matrix when updating objective function. Highly unexpected.\n" );
					//return -1;
					(this->*wide_kernel_function)( selectedI, 0, numberOfVectors-1, NULL );
					if ( 0 != gpuQCache.CheckCache( selectedI, &q_i ) )
					{
						// THIS IS VERY SERIOUS
						fprintf( stderr, "ERROR: VECTOR IS STILL NOT IN CACHE, PANICKING: %i\n", selectedI );
						return -1;
					}
				}
			}
			// create buffers for y, alpha status and QD
			{
				// calculate the work group sizes
				int remainder = activeSize % IDEAL_WORK_GROUP_SIZE;
				globalWorkSize[0] = activeSize + IDEAL_WORK_GROUP_SIZE - remainder;
				numberOfWorkGroups = globalWorkSize[0] / IDEAL_WORK_GROUP_SIZE;
				// create y buffer
				// TODO: Allocate this once
				yGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(schar) * globalWorkSize[0], NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING SPACE FOR Y DATA\n" );
					return -1;
				}
				// create alpha status buffer
				alphaStatusGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(char) * globalWorkSize[0], NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING SPACE FOR ALPHA STATUS\n" );
					return -1;
				}
				// create QD buffer
				qdGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * globalWorkSize[0], NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal witht his error
					fprintf( stderr, "ERROR CREATING SPACE FOR QD\n" );
					return -1;
				}
			}
			// write out values for y, alpha status and QD
			{
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, yGpu, CL_FALSE, 0, sizeof(schar) * activeSize, y, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING Y TO GPU WHILE SELECTING J\n" );
					return -1;
				}
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, alphaStatusGpu, CL_FALSE, 0, sizeof(char) * activeSize, alphaStatus, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING ALPHA STATUS TO GPU WHILE SELECTING J\n" );
					return -1;
				}
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, qdGpu, CL_FALSE, 0, sizeof(double) * activeSize, QD, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING QD TO GPU WHILE SELECTING J\n" );
					return -1;
				}
			}
			// wait for all writes to finish
			{
				errorCode = clFinish( kernelCommandQueue );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR FINISHING WRITES WHILE SELECTING J\n" );
					return -1;
				}
			}
			// create index and value buffers (for output)
			{
				// two buffers, one for indices, one for values
				valueBuffer = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * numberOfWorkGroups, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING SPACE FOR VALUE BUFFER WHILE SELECTING J\n" );
					return -1;
				}
				indexBuffer = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(int) * numberOfWorkGroups, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING SPACE FOR INDEX BUFFER WHILE SELECTING J\n" );
					return -1;
				}
				gMaxGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * numberOfWorkGroups, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING SPACE FOR GMAX BUFFER WHILE SELECTING J\n" );
					return -1;
				}
			}
			// set up call to kernel
			{
				// function will be of the form: kernel( y, alpha_status, QD, selectedI, Gmax, active_size, __out indices, __out values, __local scratchIndices, __local scratchValues )
				errorCode = clSetKernelArg( findCandidateJValuesKernel, 0, sizeof(cl_mem), &yGpu );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 1, sizeof(cl_mem), &alphaStatusGpu );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 2, sizeof(cl_mem), &qdGpu );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 3, sizeof(int), &selectedI );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 4, sizeof(double), &gMax );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 5, sizeof(int), &activeSize );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 6, sizeof(cl_mem), &indexBuffer );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 7, sizeof(cl_mem), &valueBuffer );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 8, sizeof(int) * IDEAL_WORK_GROUP_SIZE, NULL );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 9, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 10, sizeof(cl_mem), &gMaxGpu );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 11, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 12, sizeof(cl_mem), &q_i );
				errorCode |= clSetKernelArg( findCandidateJValuesKernel, 13, sizeof(cl_mem), &g );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR SETTING KERNEL ARGUMENTS WHILE SELECTING J\n" );
					return -1;
				}
				// set up work dimensions
				workDimension = 1;
				localWorkSize[0] = IDEAL_WORK_GROUP_SIZE;
			}
			// call the kernel
			{
				errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, findCandidateJValuesKernel, workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR RUNNING KERNEL TO FIND J CANDIDATES\n" );
					switch( errorCode )
					{
						case CL_INVALID_PROGRAM_EXECUTABLE:
							fprintf( stderr, "CL_INVALID_PROGRAM_EXECUTABLE\n" );
							break;
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "CL_INVALID_COMMAND_QUEUE\n" );
							break;
						case CL_INVALID_KERNEL:
							fprintf( stderr, "CL_INVALID_KERNEL\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_KERNEL_ARGS:
							fprintf( stderr, "CL_INVALID_KERNEL_ARGS\n" );
							break;
						case CL_INVALID_WORK_DIMENSION:
							fprintf( stderr, "CL_INVALID_WORK_DIMENSION\n" );
							break;
						case CL_INVALID_WORK_GROUP_SIZE:
							fprintf( stderr, "CL_INVALID_WORK_GROUP_SIZE\n" );
							break;
						case CL_INVALID_WORK_ITEM_SIZE :
							fprintf( stderr, "CL_INVALID_WORK_ITEM_SIZE \n" );
							break;
						case CL_INVALID_GLOBAL_OFFSET:
							fprintf( stderr, "CL_INVALID_GLOBAL_OFFSET\n" );
							break;
						case CL_OUT_OF_RESOURCES:
							fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
					};
					return -1;
				}
				errorCode = clFinish( kernelCommandQueue );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR FINISHING KERNEL WHILE SELECTING J CANDIDATES\n" );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
					};
					return -1;
				}
			}
			// read the index and value buffers
			{
				// make space on the CPU for them
				cpuValues = (double*) malloc( sizeof(double) * numberOfWorkGroups );
				if ( NULL == cpuValues )
				{
					// TODO: Deal with this for real
					fprintf( stderr, "ERROR ALLOCATING CPU SPACE FOR VALUE BUFFER\n" );
					return -1;
				}
				cpuIndices = (int*) malloc( sizeof(int) * numberOfWorkGroups );
				if ( NULL == cpuIndices )
				{
					// TODO: Deal with this for real
					fprintf( stderr, "ERROR ALLOCATING CPU SPACE FOR INDEX BUFFER\n" );
					return -1;
				}
				gmax2Candidates = (double*) malloc( sizeof(double) * numberOfWorkGroups );
				if ( NULL == gmax2Candidates )
				{
					// TODO: Deal with this for real
					fprintf( stderr, "ERROR ALLOCATING CPU SPACE FOR GMAX2 BUFFER\n" );
					return -1;
				}
				// read them
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, valueBuffer, CL_FALSE, 0, sizeof(double) * numberOfWorkGroups, cpuValues, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR READING VALUE BUFFER\n" );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, indexBuffer, CL_FALSE, 0, sizeof(int) * numberOfWorkGroups, cpuIndices, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR READING INDEX BUFFER\n" );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, gMaxGpu, CL_FALSE, 0, sizeof(double) * numberOfWorkGroups, gmax2Candidates, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR READING GMAX2 BUFFER\n" );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
			}
			// finish reads
			{
				errorCode = clFinish( kernelCommandQueue );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR: CL FINISH FAILED\n" );
					return -1;
				}
			}
			// reduce on CPU
			{
				int minIndex = -1;
				double minValue = INF;
				double maxGValue = -INF;
				for ( i = 0; i < numberOfWorkGroups; i++ )
				{
					if ( cpuValues[ i ] < minValue )
					{
						minIndex = cpuIndices[ i ];
						minValue = cpuValues[ i ];
					}
					if ( gmax2Candidates[ i ] > maxGValue )
					{
						maxGValue = gmax2Candidates[ i ];
					}
				}
				out_j = minIndex;
				out_gMax2 = maxGValue;
			}
			// get G[i] and G[j]
			{
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, g, CL_FALSE, sizeof(double) * selectedI, sizeof(double), &(outG[selectedI]), 0, NULL, NULL );
				errorCode |= clEnqueueReadBuffer( kernelCommandQueue, g, CL_FALSE, sizeof(double) * out_j, sizeof(double), &(outG[out_j]), 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR READING G[I] AND G[J] FROM GPU\n" );
					return -1;
				}
				clFinish( kernelCommandQueue );
			}
			
			// clean up
			{
				clReleaseMemObject( qdGpu );
				clReleaseMemObject( yGpu );
				clReleaseMemObject( alphaStatusGpu );
				clReleaseMemObject( gMaxGpu );
				free( cpuValues );
				free( cpuIndices );
				free( gmax2Candidates );
			}
			return 0;
		}
		
		// TODO: Make this fast
		int swap_q_indices( int i, int j )
		{
			// variables
			cl_int errorCode;
			cl_mem q_i;
			uint32_t k;
			profileDecls;
			
			// function bdy
			startTimer();
			// clear the queue, for safety
			{
				clFinish( kernelCommandQueue );
			}
			stopTimer();
			swappingOtherFunctionTime += calculateTime();
			startTimer();
			// first do regular cache swap for Q cache
			{
				if ( 0 != gpuQCache.SwapIndices( i, j ) )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR SWAPPING Q INDICES\n" );
					return -1;
				}
			}
			stopTimer();
			qCacheSwappingTime += calculateTime();
			// then swap all the individual elements
			{
				// keep the number 1
				size_t one = 1;
				// for each element the cache might hold
				for ( k = 0; k < gpuQCache.maxIndex; k++ )
				{
					// if it is cached
					startTimer();
					if ( 0 == gpuQCache.CheckCache( k, &q_i ) )
					{
						stopTimer();
						qRetrievingTime += calculateTime();
						// set up call to swap kernel
						startTimer();
						errorCode = clSetKernelArg( swapObjectiveFunctionKernel, 0, sizeof(cl_mem), &q_i );
						errorCode |= clSetKernelArg( swapObjectiveFunctionKernel, 1, sizeof(int), &i );
						errorCode |= clSetKernelArg( swapObjectiveFunctionKernel, 2, sizeof(int), &j );
						stopTimer();
						argSettingTime += calculateTime();
						if ( CL_SUCCESS != errorCode )
						{
							// TODO: Deal with this error
							fprintf( stderr, "ERROR SETTING ARGUMENTS FOR Q SWAPPING\n" );
							return -1;
						}
						// call swap kernel
						startTimer();
						errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, swapObjectiveFunctionKernel, 1, NULL,
															&one, &one, 0, NULL, NULL );
						stopTimer();
						kernelEnqueuingTime += calculateTime();
						if (CL_SUCCESS != errorCode )
						{
							// TODO: Deal with this error
							fprintf( stderr, "ERROR RUNNING Q SWAPPING KERNEL\n" );
							return -1;
						}
					}
					else
					{
						stopTimer();
						qRetrievingTime += calculateTime();
					}
				}
			}
			// try this without calling a kernel
			/*{
				cl_mem * qValues = (cl_mem*) malloc( sizeof(cl_mem) * gpuQCache.maxIndex );
				double * copiedValues = (double*) malloc( sizeof(double) * 2 * gpuQCache.maxIndex );
				int qIndex = 0;
				
				// for each element the cache might hold
				for ( k = 0; k < gpuQCache.maxIndex; k++ )
				{
					// if it is cached
					startTimer();
					if ( 0 == gpuQCache.CheckCache( k, &q_i ) )
					{
						stopTimer();
						qRetrievingTime += calculateTime();
						startTimer();
						qValues[ qIndex ] = q_i;
						errorCode = clEnqueueReadBuffer( kernelCommandQueue, q_i, CL_FALSE, sizeof(double) * i, sizeof(double), &(copiedValues[ 2 * qIndex ]), 0, NULL, NULL );
						if ( CL_SUCCESS != errorCode )
						{
							fprintf( stderr, "ERROR READING I VALUES WHILE Q SWAPPING\n" );
							return -1;
						}
						errorCode = clEnqueueReadBuffer( kernelCommandQueue, q_i, CL_FALSE, sizeof(double) * j, sizeof(double), &(copiedValues[ 2 * qIndex + 1 ]), 0, NULL, NULL );
						if ( CL_SUCCESS != errorCode )
						{
							fprintf( stderr, "ERROR READING J VALUES WHILE Q SWAPPING\n" );
							return -1;
						}
						qIndex++;
						stopTimer();
						kernelEnqueuingTime += calculateTime();
					}
					else
					{
						stopTimer();
						qRetrievingTime += calculateTime();
					}
				}
				startTimer();
				clFinish( kernelCommandQueue );
				stopTimer();
				kernelEnqueuingTime += calculateTime();
				for ( k = 0; k < qIndex; k++ )
				{
					startTimer();
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, qValues[ k ], CL_FALSE, sizeof(double) * j, sizeof(double), &(copiedValues[ 2 * k ]), 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						fprintf( stderr, "ERROR WRITING I VALUES WHILE Q SWAPPING\n" );
						return -1;
					}
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, qValues[ k ], CL_FALSE, sizeof(double) * i, sizeof(double), &(copiedValues[ 2 * k + 1 ]), 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						fprintf( stderr, "ERROR WRITING J VALUES WHILE Q SWAPPING\n" );
						return -1;
					}
					stopTimer();
					argSettingTime += calculateTime();
				}
				startTimer();
				clFinish( kernelCommandQueue );
				stopTimer();
				argSettingTime += calculateTime();
				free( qValues );
				free( copiedValues );
			}
			*/
			startTimer();
			// clear the queue
			{
				clFinish( kernelCommandQueue );
			}
			stopTimer();
			endQueueTime += calculateTime();
			
			//stopTimer();
			//swappingTime += calculateTime();
			swappingCount++;
			
			// clean up
			return 0;
		}
		
		int swap_vector_block_indices( int i, int j )
		{
			// variables
			cl_int errorCode;
			cl_mem A;
			int workDimension;
			size_t globalWorkSize[3];
			size_t localWorkSize[3];
			
			// function body
			// clear queue, for safety
			{
				clFinish( kernelCommandQueue );
			}
			// conjure up cached memory
			{
				// if we can't get it
				if ( 0 != gpuCache.GetASpace( &A ) )
				{
					// error!!!
					//fprintf( stderr, "ERROR, NO A IS CACHED\n" );
					//return -1;
					return 0;
				}
			}
			// set up call to swapping kernel
			{
				// set up work dimensions
				workDimension = 1;
				globalWorkSize[0] = numberOfVectors + (IDEAL_WORK_GROUP_SIZE - (numberOfVectors % IDEAL_WORK_GROUP_SIZE));
				localWorkSize[0] = IDEAL_WORK_GROUP_SIZE;
				// set arguments
				// call will be of the form: kernel( A, rows, cols, i, j )
				errorCode = clSetKernelArg( swapVectorBlockKernel, 0, sizeof(cl_mem), &A );
				errorCode |= clSetKernelArg( swapVectorBlockKernel, 1, sizeof(int), &numberOfVectors );
				errorCode |= clSetKernelArg( swapVectorBlockKernel, 2, sizeof(int), &(x[0].dim) );
				errorCode |= clSetKernelArg( swapVectorBlockKernel, 3, sizeof(int), &(i) );
				errorCode |= clSetKernelArg( swapVectorBlockKernel, 4, sizeof(int), &(j) );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR SETTING VECTOR BLOCK SWAP ARGUMENTS\n" );
					return -1;
				}
			}
			// call swapping kernel
			{
				// just call it
				errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, swapVectorBlockKernel, workDimension, 
													NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR ENQUEUING SWAP VECTOR BLOCK KERNEL\n" );
					return -1;
				}
			}
			
			// clean up
			return 0;
		}
		
	#endif
	
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

	int (Kernel::*wide_kernel_function)( int i, int startJ, int endJ, double * output ) const; 
	
private:

	// data members
	
#ifdef _DENSE_REP
	svm_node *x;
#else
	const svm_node **x;
#endif
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	// OpenCL related data members
	cl_context kernelContext;
	cl_context kernelCpuContext;
	cl_command_queue kernelCommandQueue;
	cl_command_queue kernelCpuCommandQueue;
	// void( global double * x_data_i, global double * x_data_j, global double * outputData, int goodDataSize, local double * scratchData );
	cl_kernel linearKernelKernel;
	cl_kernel customDaxpyKernel;
	cl_kernel swapObjectiveFunctionKernel;
	cl_kernel findCandidateIValuesKernel;
	cl_kernel findCandidateJValuesKernel;
	cl_kernel dualDaxpyKernel;
	cl_kernel customMatrixVectorKernel;
	cl_kernel customMatrixVectorPolynomialKernel;
	cl_kernel customMatrixVectorSigmoidKernel;
	cl_kernel customMatrixVectorRBFKernel;
	cl_kernel reductionKernel;
	cl_kernel swapVectorBlockKernel;
	cl_kernel predictionReductionKernel;
	cl_mem resultCl;
	// caching stuff
	GPUCache gpuCache;
	GPUCache gpuQCache;
	GPUCache cpuQCache;
	const char * linearKernelKernelSource;// = 
	//#include "linearKernelKernelSource.cl"
	;
	const char * customDaxpyKernelSource// = 
	//#include "customDaxpyKernelSource.cl"
	;
	const char * swapObjectiveFunctionKernelSource// = 
	//#include "swapObjectiveFunctionKernelSource.cl"
	;
	const char * findCandidateIValuesKernelSource// = 
	//#include "findCandidateIValuesKernelSource.cl"
	;
	const char * findCandidateJValuesKernelSource// = 
//	#include "findCandidateJValuesKernelSource.cl"
	;
	const char * dualDaxpyKernelSource// =
//	#include "dualDaxpyKernelSource.cl"
	;
	const char * customMatrixVectorKernelSource// = 
//	#include "customMatrixVectorKernelSource.cl"
	;
	const char * reductionKernelSource// = 
//	#include "reductionKernelSource.cl"
	;
	const char * swapVectorBlockKernelSource// =
	//#include "swapVectorBlockKernelSource.cl"
	;
	const char * customMatrixVectorPolynomialKernelSource// =
	//#include "customMatrixVectorPolynomialKernelSource.cl"
	;
	const char * customMatrixVectorSigmoidKernelSource// =
	//#include "customMatrixVectorSigmoidKernelSource.cl"
	;
	const char * customMatrixVectorRBFKernelSource// =
//	#include "customMatrixVectorRBFKernelSource.cl"
	;
	const char * predictionReductionKernelSource// = 
//	#include "predictionReductionKernelSource.cl"
	;
	cl_mem x_data_j;
	
	// function declarations
	
	static double dot(const svm_node *px, const svm_node *py);
#ifdef _DENSE_REP
	static double dot(const svm_node &px, const svm_node &py);
#endif

	// function code

	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
#ifdef _DENSE_REP
		return (x+i)->values[(int)((x+j)->values[0])];
#else
		return x[i][(int)(x[j][0].value)].value;
#endif
	}
	
	// let's create a collection of parallel kernels that we can call like all the other
	// kernels
	double kernel_linear_opencl( int i, int j ) const
	{
	
		// debugging
		//fprintf( stdout, "Being called\n" );
	
		// variables
		double result;
		double * longResult;
		int k;
		uint32_t workGroupSize;
		//int goodDataSize;
		uint32_t goodDataSize;
		clAmdBlasStatus status;
		cl_mem x_data_i;
		cl_mem x_data_j;
		cl_mem y_data;
		//cl_mem resultCl;
		cl_int errorCode;
		cl_uint workDimension;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		size_t numberOfRows;
		size_t numberOfColumns;
		// profiling
		LARGE_INTEGER startCount;
		LARGE_INTEGER endCount;
		LONGLONG writeTime, setupTime, executeTime, readTime;
		LONGLONG totalTime;
		cl_event write1Event;
		cl_event write2Event;
		cl_event readEvent;
		cl_event executeEvent;
		cl_ulong write1Start, write1End, write2Start, write2End, readStart, readEnd, executeStart, executeEnd;
		
		
		// function body
		QueryPerformanceCounter( &startCount );
		// push the two vectors to the GPU
		{
			#ifdef _DENSE_REP
				// allocate GPU space for them
				// check the cache
				if ( -1 == ((GPUCache)gpuCache).CheckCache( i, &x_data_i ) )
				{
					x_data_i = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * x[i].dim, NULL, &errorCode );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this error case
						fprintf( stderr, "Error creating OpenCL buffer\n" );
						exit( -1 );
					}
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, x_data_i, /* CL_FALSE /*/ CL_TRUE /**/, 0, sizeof(double) * x[i].dim, x[i].values, 0, NULL, &write1Event );
					((GPUCache)gpuCache).CacheData( i, x_data_i );
				}
				if ( -1 == ((GPUCache)gpuCache).CheckCache( j, &x_data_j ) )
				{
					x_data_j = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * x[j].dim, NULL, &errorCode );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this error code
						fprintf( stderr, "Error creating OpenCL buffer\n" );
						exit( -1 );
					}
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, x_data_j, /* CL_FALSE /*/ CL_TRUE /**/, 0, sizeof(double) * x[j].dim, x[j].values, 0, NULL, &write2Event );
					((GPUCache)gpuCache).CacheData( j, x_data_j );
				}
				
				// do the actual writing
				//errorCode = clEnqueueWriteBuffer( kernelCommandQueue, x_data_i, CL_TRUE, 0, sizeof(double) * x[i].dim, x[i].values, 0, NULL, NULL );
				//errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, x_data_j, CL_TRUE, 0, sizeof(double) * x[j].dim, x[j].values, 0, NULL, NULL );
				/*if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "Error writing to OpenCL buffers\n" );
					exit( -1 );
				}*/
			#else
				// TODO: this
			#endif
			// create space for the output
			/*resultCl = clCreateBuffer( kernelContext, CL_MEM_WRITE_ONLY, sizeof(double), NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "Error creating linear kernel result buffer\n" );
				exit( -1 );
			}*/
			// debugging
			//fprintf( stdout, "Created buffers successfully\n" );
		}
		QueryPerformanceCounter( &endCount );
		writeTime = endCount.QuadPart - startCount.QuadPart;
		// set up the openCL kernel
		// we will not be using clAmdBlas
		{
			// set up dimensions and work group sizes
			// set the work group size to the next power of 2 after the largest value that is necessary (because we need to do reduction)
			goodDataSize = min( x[i].dim, x[j].dim );
			workGroupSize = goodDataSize;
			/*workGroupSize--;
			workGroupSize |= workGroupSize >> 1;
			workGroupSize |= workGroupSize >> 2;
			workGroupSize |= workGroupSize >> 4;
			workGroupSize |= workGroupSize >> 8;
			workGroupSize |= workGroupSize >> 16;
			workGroupSize++;*/
			// allocate space for intermediate GPU result, we'll do reduction on the CPU
			y_data = clCreateBuffer( kernelContext, CL_MEM_WRITE_ONLY, sizeof(double) * goodDataSize, NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error code
				fprintf( stderr, "Error creating OpenCL buffer\n" );
				exit( -1 );
			}
			// debugging
			//fprintf( stdout, "Work Group size: %i\n", workGroupSize );
			//workGroupSize = (int) ( pow(2, ceil(log(goodDataSize)/log(2))) );
			workDimension = 1;
			globalWorkSize[0] = workGroupSize;
			localWorkSize[0] = workGroupSize;
			// set the arguments
			errorCode = clSetKernelArg( linearKernelKernel, 0, sizeof(cl_mem), &(x_data_i) );
			errorCode |= clSetKernelArg( linearKernelKernel, 1, sizeof(cl_mem), &(x_data_j) );
			//errorCode |= clSetKernelArg( linearKernelKernel, 2, sizeof(cl_mem), &(resultCl) );
			errorCode |= clSetKernelArg( linearKernelKernel, 2, sizeof(cl_mem), &(y_data) );
			errorCode |= clSetKernelArg( linearKernelKernel, 3, sizeof(int), &(goodDataSize) );
			errorCode |= clSetKernelArg( linearKernelKernel, 4, sizeof(double) * workGroupSize, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error for real
				fprintf( stderr, "Error setting kernel arguments\n" );
				exit( -1 );
			}
			// debugging
			//fprintf( stdout, "Set up kernel successfully\n" );
		}
		QueryPerformanceCounter( &startCount );
		setupTime = startCount.QuadPart - endCount.QuadPart;
		// execute it
		{
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, linearKernelKernel, workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, &executeEvent );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: deal with this error
				fprintf( stderr, "Error running linear kernel\n" );
				exit( -1 );
			}
			clFinish( kernelCommandQueue );
			// debugging
			//fprintf( stdout, "Executed kernel successfully\n" );
		}
		QueryPerformanceCounter( &endCount );
		executeTime = endCount.QuadPart - startCount.QuadPart;
		// we will not be using clAmdBlas because we can't control that it wants a const command queue, which we can't give it
		/*{
			numberOfRows = 1;
			numberOfColumns = min(x[i].dim, x[j].dim);
			status = clAmdBlasDgemv( clAmdBlasRowMajor, clAmdBlasNoTrans, numberOfRows, numberOfColumns, 1.0, x_data_i, numberOfColumns, x_data_j, 0, 1, 0.0, resultCl, 
									 0, 1, 1, &kernelCommandQueue, 0, NULL, NULL );
		}*/
		// read the result from the GPU
		{
			//errorCode = clEnqueueReadBuffer( kernelCommandQueue, resultCl, /* CL_FALSE /*/ CL_TRUE /***/, 0, sizeof(double), &result, 0, NULL, &readEvent );
			/*if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "Error reading linear kernel result\n" );
				exit( -1 );
			}*/
			//clFinish( kernelCommandQueue );
			longResult = (double*) malloc( sizeof(double) * goodDataSize );
			if ( NULL == longResult )
			{
				fprintf( stderr, "ERROR ALLOCATING LONG RESULT\n" );
				exit( -1 );
			}
			//double longResult[ goodDataSize ];
			errorCode = clEnqueueReadBuffer( kernelCommandQueue, y_data, /*CL_FALSE*/CL_TRUE, 0, sizeof(double) * goodDataSize, longResult, 0, NULL, &readEvent );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "Error reading linear kernel result\n" );
				exit( -1 );
			}
			// debugging
			//fprintf( stdout, "Read back data successfully\n" );
			//clFinish( kernelCommandQueue );
			result = 0.0;
			for ( k = 0; k < goodDataSize; k++ )
			{
				result = result + longResult[ k ];
			}
		}
		QueryPerformanceCounter( &startCount );
		readTime = startCount.QuadPart - endCount.QuadPart;
		
		// report profiling
		totalTime = writeTime + setupTime + executeTime + readTime;
		fprintf( stdout, "Write time: %lf\n", (double)writeTime/((double)totalTime) );
		fprintf( stdout, "Setup time: %lf\n", (double)setupTime/((double)totalTime) );
		fprintf( stdout, "Execution time: %lf\n", (double)executeTime/((double)totalTime) );
		fprintf( stdout, "Read time: %lf\n", (double)readTime/((double)totalTime) );
		fprintf( stdout, "Total time: %llu\n", totalTime );
		fprintf( stdout, "Exe time: %llu\n", executeTime );
	
		clGetEventProfilingInfo( write1Event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write1Start, NULL);
		clGetEventProfilingInfo( write1Event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write1End, NULL);
		clGetEventProfilingInfo( write2Event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write2Start, NULL);
		clGetEventProfilingInfo( write2Event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write2End, NULL);
		clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &readStart, NULL);
		clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &readEnd, NULL);
		clGetEventProfilingInfo( executeEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &executeStart, NULL);
		clGetEventProfilingInfo( executeEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &executeEnd, NULL);
		fprintf( stdout, "Write 1 time: %llu\n", (write1End - write1Start) );
		fprintf( stdout, "Write 2 time: %llu\n", (write2End - write2Start) );
		fprintf( stdout, "Read time: %llu\n", (readEnd - readStart) );
		fprintf( stdout, "Execute time: %llu\n", (executeEnd - executeStart) );
		
		// clean up
		//clReleaseMemObject( x_data_i );
		//clReleaseMemObject( x_data_j );
		//clReleaseMemObject( resultCl );
		clReleaseMemObject( y_data );
		free( longResult );
		return result;
	}
	
	// this will be a set of special, "wide" kernels
	
	#ifdef CL_SVM
	
	/**
	 *	wide_kernel_linear_opencl
	 *
	 *	This will perform the dot product between the ith value and the startJ through endJth values
	 *	simultaenously
	 **/
	int wide_kernel_linear_opencl( int i, int startJ, int endJ, double * output ) const
	{
	
		// debugging
		//fprintf( stdout, "RUNNING LINEAR KERNEL\n" );
	
		// variables
		clAmdBlasStatus blasStatus;
		cl_int errorCode;
		cl_mem x_data_i;
		cl_mem x_data_j;
		cl_mem y_data;
		cl_mem tempBuffer;
		//cl_mem q_i;
		uint32_t jDimension;
		uint32_t k;
		int numberOfJVectors;
		int qAlreadyComputed;
		int workDimension;
		int numberOfWorkGroups;
		int retried;
		double * tempData;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		profileDecls;
		GPUCache * myGpuCache;
		GPUCache * myCpuCache;
	
		// function body
		myGpuCache = (GPUCache*)(&gpuCache);
		myCpuCache = (GPUCache*)(&cpuQCache);
		retried = 0;
		// see if we've already done the work
		{
			// calculate the number of vectors we need (just in case)
			numberOfJVectors = (endJ - startJ) + 1;
			qAlreadyComputed = 0;
			// all programming is an exercise in caching
			if ( 0 == ((GPUCache)gpuQCache).CheckCache( i, &y_data ) )
			{
				qAlreadyComputed = 1;
				// I really, sincerely don't care that gotos are bad practice
				goto wide_kernel_linear_opencl_skip_work_label;
			}
		}
wide_kernel_linear_opencl_retry_label:
		startTimer();
		// write the ith vector out to the GPU
		{	
			#ifdef _DENSE_REP
				//fprintf( stdout, "WHATWHAT\n" );
				//if ( -1 == ((GPUCache)gpuCache).CheckCache( i, &x_data_i ) )
				if ( -1 == myGpuCache->CheckCache( i, &x_data_i ) )
				{
					x_data_i = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE/*ONLY*/, sizeof(double) * x[i].dim, NULL, &errorCode );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this
						fprintf( stderr, "ERROR CREATING X_DATA_I BUFFER\n" );
						return -1;
					}
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, x_data_i, CL_FALSE, 0, sizeof(double) * x[i].dim, x[i].values, 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this error
						fprintf( stderr, "ERROR WRITING X DATA TO BUFFER\n" );
						switch( errorCode )
						{
							case CL_MEM_OBJECT_ALLOCATION_FAILURE:
								fprintf( stderr, "RAN OUT OF MEMORY\n" );
								if ( !retried )
								{
									fprintf( stderr, "CLEARING MEMORY AND THEN RETRYING\n" );
									retried = 1;
									//gpuCache.InvalidateCache();
									myGpuCache->InvalidateCache();
									//cpuQCache.InvalidateCache();
									myCpuCache->InvalidateCache();
									goto wide_kernel_linear_opencl_retry_label;
								}
								else
								{
									fprintf( stderr, "ALREADY TRIED CLEARING MEMORY, GIVING UP\n" );
								}
								break;
							case CL_INVALID_COMMAND_QUEUE:
								fprintf( stderr, "CL_INVALID_COMMAND_QUEUE\n" );
								break;
							case CL_INVALID_CONTEXT:
								fprintf( stderr, "CL_INVALID_CONTEXT\n" );
								break;
							case CL_INVALID_MEM_OBJECT:
								fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
								break;
							case CL_INVALID_VALUE:
								fprintf( stderr, "CL_INVALID_VALUE\n" );
								break;
							case CL_INVALID_EVENT_WAIT_LIST:
								fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
								break;
							case CL_OUT_OF_HOST_MEMORY:
								fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
								break;
							default:
								fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
								break;
						};
						return -1;
					}
					//fprintf( stdout, "WHO?\n" );
					//((GPUCache)gpuCache).CacheData( i, x_data_i );
					myGpuCache->CacheData( i, x_data_i );
				}
			#else
				// TODO: Handle nondense rep
			#endif
		}
		// write the other vectors out to the GPU
		{
			jDimension = x[startJ].dim;
			//if ( -1 == gpuCache.CheckCache( startJ, endJ, &x_data_j ) )
			if ( -1 == myGpuCache->CheckCache( startJ, endJ, &x_data_j ) )
			{
				// create the cl_mem buffer (assume all j vectors have same dimensionality)
				x_data_j = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfJVectors * jDimension, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING BUFFER FOR J VALUES\n" );
					return -1;
				}
				// loop through the j vectors and write them
				errorCode = 0;
				for ( k = 0; k < numberOfJVectors; k++ )
				{
					errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, x_data_j, CL_FALSE, k * sizeof(double) * jDimension, sizeof(double) * jDimension, x[k + startJ].values, 0, NULL, NULL );
				}
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR WRITING J VALUES TO GPU MEMORY\n" );
					return -1;
				}
				//gpuCache.CacheData( startJ, endJ, x_data_j );
				myGpuCache->CacheData( startJ, endJ, x_data_j );
			}
		}
		// wait for all the writing to finish
		{
			errorCode = clFinish( kernelCommandQueue );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this
				fprintf( stderr, "ERROR FINISHING WRITING EVENTS\n" );
				return -1;
			}
		}
		stopTimer();
		writingVectorsTime += calculateTime();
		startTimer();
		// set up the output
		{
			int remainder = numberOfJVectors % IDEAL_WORK_GROUP_SIZE;
			//if ( -1 == gpuCache.GetQSpace( numberOfJVectors, &y_data ) )
			//if( -1 == gpuQCache.CheckCache( i, &y_data ) )
			{
				y_data = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * (numberOfJVectors + IDEAL_WORK_GROUP_SIZE - remainder), NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING Y DATA BUFFER\n" );
					return -1;
				}
				//gpuQCache.CacheData( i, y_data );
			}
		}
		stopTimer();
		outputSetupTime += calculateTime();
		startTimer();
		// set up call to custom kernel
		{
			int remainder = jDimension % IDEAL_WORK_GROUP_SIZE;
			int workingSize = jDimension + IDEAL_WORK_GROUP_SIZE - remainder;
			numberOfWorkGroups = workingSize / IDEAL_WORK_GROUP_SIZE;
			int localSize = IDEAL_WORK_GROUP_SIZE;
			// set up work group dimensions
			workDimension = 2;
			globalWorkSize[0] = numberOfJVectors;
			globalWorkSize[1] = IDEAL_WORK_GROUP_SIZE; //workingSize;
			localWorkSize[0] = 1;
			localWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
			//tempBuffer = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * ( numberOfWorkGroups ) * numberOfJVectors, NULL, &errorCode );
			//tempData = (double*) malloc( sizeof(double) * ( numberOfWorkGroups ) * numberOfJVectors );
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR CREATING TEMP BUFFER\n" );
				return -1;
			}
			// set up arguments
			// call will be of the form kernel( A, x, y, rows, cols, __local scratch )
			errorCode = clSetKernelArg( customMatrixVectorKernel, 0, sizeof(cl_mem), &x_data_j );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 1, sizeof(cl_mem), &x_data_i );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 2, sizeof(cl_mem), &y_data );
			//errorCode |= clSetKernelArg( customMatrixVectorKernel, 3, sizeof(int), &numberOfJVectors );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 3, sizeof(int), &jDimension );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 4, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			//errorCode |= clSetKernelArg( customMatrixVectorKernel, 6, sizeof(cl_mem), &tempBuffer );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 5, sizeof(int), &localSize );
			//errorCode |= clSetKernelArg( customMatrixVectorKernel, 8, sizeof(int), &numberOfWorkGroups );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING KERNEL ARGS FOR CUSTOM GEMV\n" );
				return -1;
			}
		}
		// call our custom kernel (alternative to cl blas library)
		retried = 0;
		{
			// debugging
			//fprintf( stderr, "ABOUT TO RUN CUSTOM KERNEL\n" );
			// y = Ax
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, customMatrixVectorKernel, workDimension,
												NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING CUSTOM MATRIX VECTOR KERNEL\n");
				switch( errorCode )
				{
					case CL_INVALID_PROGRAM_EXECUTABLE:
						fprintf( stderr, "CL_INVALID_PROGRAM_EXECUTABLE\n" );
						break;
					case CL_INVALID_COMMAND_QUEUE :
						fprintf( stderr, "CL_INVALID_COMMAND_QUEUE \n" );
						break;
					case CL_INVALID_KERNEL:
						fprintf( stderr, "CL_INVALID_KERNEL\n" );
						break;
					case CL_INVALID_CONTEXT:
						fprintf( stderr, "CL_INVALID_CONTEXT\n" );
						break;
					case CL_INVALID_KERNEL_ARGS:
						fprintf( stderr, "CL_INVALID_KERNEL_ARGS\n" );
						break;
					case CL_INVALID_WORK_DIMENSION:
						fprintf( stderr, "CL_INVALID_WORK_DIMENSION\n" );
						break;
					case CL_INVALID_WORK_GROUP_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_GROUP_SIZE: %i, %i\n", jDimension, CL_DEVICE_MAX_WORK_GROUP_SIZE );
						break;
					case CL_INVALID_WORK_ITEM_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_ITEM_SIZE\n" );
						break;
					case CL_INVALID_GLOBAL_OFFSET:
						fprintf( stderr, "CL_INVALID_GLOBAL_OFFSET\n" );
						break;
					case CL_OUT_OF_RESOURCES:
						fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
						break;
					case CL_MEM_OBJECT_ALLOCATION_FAILURE:
						fprintf( stderr, "RAN OUT OF MEMORY\n" );
						if ( !retried )
						{
							fprintf( stderr, "CLEARING MEMORY AND THEN RETRYING\n" );
							retried = 1;
							//gpuCache.InvalidateCache();
							myGpuCache->InvalidateCache();
							//cpuQCache.InvalidateCache();
							myCpuCache->InvalidateCache();
							goto wide_kernel_linear_opencl_retry_label;
						}
						else
						{
							fprintf( stderr, "ALREADY TRIED CLEARING MEMORY, GIVING UP\n" );
						}
						break;
					case CL_INVALID_EVENT_WAIT_LIST:
						fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
						break;
					case CL_OUT_OF_HOST_MEMORY:
						fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
						break;
					default:
						fprintf( stderr, "OTHER FAILURE: %i\n", errorCode );
						break;
				};
				return -1;
			}
			// debugging
			//fprintf( stdout, "Call to enqueue custom GEMV went well\n" );
			clFinish( kernelCommandQueue );
		}
		stopTimer();
		blasExecutionTime += calculateTime();
		// read the output
		// I really, sincerely don't care that gotos are bad practice
wide_kernel_linear_opencl_skip_work_label:
		startTimer();
		{
			if ( NULL != output )
			{
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, y_data, CL_TRUE, 0, sizeof(double) * numberOfJVectors, output, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR READING OUTPUT DATA: %i, %i, %i\n", i, startJ, endJ );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						case CL_OUT_OF_RESOURCES:
							fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
			}
		}
		stopTimer();
		outputReadingTime += calculateTime();
		
		// clean up
		{
			//clReleaseMemObject( x_data_i );
			//clReleaseMemObject( x_data_j );
			clReleaseMemObject( y_data );
		}
		return 0;
	}
	
	int wide_kernel_poly_opencl( int i, int startJ, int endJ, double * output ) const
	{
		// variables
		clAmdBlasStatus blasStatus;
		cl_int errorCode;
		cl_mem x_data_i;
		cl_mem x_data_j;
		cl_mem y_data;
		cl_mem tempBuffer;
		//cl_mem q_i;
		uint32_t jDimension;
		uint32_t k;
		int numberOfJVectors;
		int qAlreadyComputed;
		int workDimension;
		int numberOfWorkGroups;
		int retried;
		double * tempData;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		profileDecls;
		GPUCache * myGpuCache;
		GPUCache * myCpuCache;
		GPUCache * myGpuQCache;
		
		// function body
		myGpuCache = (GPUCache*)(&gpuCache);
		myCpuCache = (GPUCache*)(&cpuQCache);
		myGpuQCache = (GPUCache*)(&gpuQCache);
		// see if we've already done the work
		{
			// calculate the number of vectors we need (just in case)
			numberOfJVectors = (endJ - startJ) + 1;
			qAlreadyComputed = 0;
			// all programming is an exercise in caching
			//if ( 0 == gpuQCache.CheckCache( i, &y_data ) )
			if ( 0 == myGpuQCache->CheckCache( i, &y_data ) )
			{
				qAlreadyComputed = 1;
				// I really, sincerely don't care that gotos are bad practice
				goto wide_kernel_poly_opencl_skip_work_label;
			}
		}
		startTimer();
		retried = 0;
wide_kernel_poly_opencl_retry_label:
		// write the ith vector out to the GPU
		{
			#ifdef _DENSE_REP
				//if ( -1 == gpuCache.CheckCache( i, &x_data_i ) )
				if ( -1 == myGpuCache->CheckCache( i, &x_data_i ) )
				{
					// debugging
					//fprintf( stdout, "MISSED CACHE ON X[%i]\n", i );
					x_data_i = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE/*ONLY*/, sizeof(double) * x[i].dim, NULL, &errorCode );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this
						fprintf( stderr, "ERROR CREATING X_DATA_I BUFFER\n" );
						return -1;
					}
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, x_data_i, CL_FALSE, 0, sizeof(double) * x[i].dim, x[i].values, 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this error
						fprintf( stderr, "ERROR WRITING X DATA TO BUFFER\n" );
						switch( errorCode )
						{
							case CL_MEM_OBJECT_ALLOCATION_FAILURE:
								fprintf( stderr, "RAN OUT OF MEMORY\n" );
								if ( !retried )
								{
									fprintf( stderr, "CLEARING MEMORY AND THEN RETRYING\n" );
									retried = 1;
									//gpuCache.InvalidateCache();
									//cpuQCache.InvalidateCache();
									myGpuCache->InvalidateCache();
									myCpuCache->InvalidateCache();
									goto wide_kernel_poly_opencl_retry_label;
								}
								else
								{
									fprintf( stderr, "ALREADY TRIED CLEARING MEMORY, GIVING UP\n" );
								}
								break;
							case CL_INVALID_COMMAND_QUEUE:
								fprintf( stderr, "CL_INVALID_COMMAND_QUEUE\n" );
								break;
							case CL_INVALID_CONTEXT:
								fprintf( stderr, "CL_INVALID_CONTEXT\n" );
								break;
							case CL_INVALID_MEM_OBJECT:
								fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
								break;
							case CL_INVALID_VALUE:
								fprintf( stderr, "CL_INVALID_VALUE\n" );
								break;
							case CL_INVALID_EVENT_WAIT_LIST:
								fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
								break;
							case CL_OUT_OF_HOST_MEMORY:
								fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
								break;
							default:
								fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
								break;
						};
						return -1;
					}
					//gpuCache.CacheData( i, x_data_i );
					myGpuCache->CacheData( i, x_data_i );
				}
			#else
				// TODO: Handle nondense rep
			#endif
		}
		// write the other vectors out to the GPU
		{
			jDimension = x[startJ].dim;
			//if ( -1 == gpuCache.CheckCache( startJ, endJ, &x_data_j ) )
			if ( -1 == myGpuCache->CheckCache( startJ, endJ, &x_data_j ) )
			{
				// debugging
				//fprintf( stdout, "MISSED CACHE ON A\n" );
				// create the cl_mem buffer (assume all j vectors have same dimensionality)
				x_data_j = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfJVectors * jDimension, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING BUFFER FOR J VALUES\n" );
					return -1;
				}
				// loop through the j vectors and write them
				errorCode = 0;
				for ( k = 0; k < numberOfJVectors; k++ )
				{
					errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, x_data_j, CL_FALSE, k * sizeof(double) * jDimension, sizeof(double) * jDimension, x[k + startJ].values, 0, NULL, NULL );
				}
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR WRITING J VALUES TO GPU MEMORY\n" );
					return -1;
				}
				//gpuCache.CacheData( startJ, endJ, x_data_j );
				myGpuCache->CacheData( startJ, endJ, x_data_j );
			}
		}
		// wait for all the writing to finish
		{
			errorCode = clFinish( kernelCommandQueue );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this
				fprintf( stderr, "ERROR FINISHING WRITING EVENTS\n" );
				return -1;
			}
		}
		stopTimer();
		writingVectorsTime += calculateTime();
		startTimer();
		// set up the output
		{
			int remainder = numberOfJVectors % IDEAL_WORK_GROUP_SIZE;
			//if ( -1 == gpuCache.GetQSpace( numberOfJVectors, &y_data ) )
			//if( -1 == gpuQCache.CheckCache( i, &y_data ) )
			{
				y_data = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * (numberOfJVectors + IDEAL_WORK_GROUP_SIZE - remainder), NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING Y DATA BUFFER\n" );
					return -1;
				}
				//gpuQCache.CacheData( i, y_data );
			}
		}
		stopTimer();
		outputSetupTime += calculateTime();
		startTimer();
		// set up call to custom kernel
		{
			int remainder = jDimension % IDEAL_WORK_GROUP_SIZE;
			int workingSize = jDimension + IDEAL_WORK_GROUP_SIZE - remainder;
			numberOfWorkGroups = workingSize / IDEAL_WORK_GROUP_SIZE;
			int localSize = IDEAL_WORK_GROUP_SIZE;
			// set up work group dimensions
			workDimension = 2;
			globalWorkSize[0] = numberOfJVectors;
			globalWorkSize[1] = IDEAL_WORK_GROUP_SIZE; //workingSize;
			localWorkSize[0] = 1;
			localWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
			//tempBuffer = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * ( numberOfWorkGroups ) * numberOfJVectors, NULL, &errorCode );
			//tempData = (double*) malloc( sizeof(double) * ( numberOfWorkGroups ) * numberOfJVectors );
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR CREATING TEMP BUFFER\n" );
				return -1;
			}
			// set up arguments
			// call will be of the form kernel( A, x, y, rows, cols, __local scratch, double gamma, double coef0, int degree )
			errorCode = clSetKernelArg( customMatrixVectorPolynomialKernel, 0, sizeof(cl_mem), &x_data_j );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 1, sizeof(cl_mem), &x_data_i );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 2, sizeof(cl_mem), &y_data );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 3, sizeof(int), &jDimension );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 4, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 5, sizeof(int), &localSize );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 6, sizeof(double), &gamma );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 7, sizeof(double), &coef0 );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 8, sizeof(int), &degree );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING KERNEL ARGS FOR CUSTOM GEMV\n" );
				return -1;
			}
		}
		// call our custom kernel (alternative to cl blas library)
		{
			// debugging
			//fprintf( stderr, "ABOUT TO RUN CUSTOM KERNEL\n" );
			// y = Ax
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, customMatrixVectorPolynomialKernel, 
												workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING CUSTOM MATRIX VECTOR KERNEL\n");
				switch( errorCode )
				{
					case CL_INVALID_PROGRAM_EXECUTABLE:
						fprintf( stderr, "CL_INVALID_PROGRAM_EXECUTABLE\n" );
						break;
					case CL_INVALID_COMMAND_QUEUE :
						fprintf( stderr, "CL_INVALID_COMMAND_QUEUE \n" );
						break;
					case CL_INVALID_KERNEL:
						fprintf( stderr, "CL_INVALID_KERNEL\n" );
						break;
					case CL_INVALID_CONTEXT:
						fprintf( stderr, "CL_INVALID_CONTEXT\n" );
						break;
					case CL_INVALID_KERNEL_ARGS:
						fprintf( stderr, "CL_INVALID_KERNEL_ARGS\n" );
						break;
					case CL_INVALID_WORK_DIMENSION:
						fprintf( stderr, "CL_INVALID_WORK_DIMENSION\n" );
						break;
					case CL_INVALID_WORK_GROUP_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_GROUP_SIZE: %i, %i\n", jDimension, CL_DEVICE_MAX_WORK_GROUP_SIZE );
						break;
					case CL_INVALID_WORK_ITEM_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_ITEM_SIZE\n" );
						break;
					case CL_INVALID_GLOBAL_OFFSET:
						fprintf( stderr, "CL_INVALID_GLOBAL_OFFSET\n" );
						break;
					case CL_OUT_OF_RESOURCES:
						fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
						break;
					case CL_MEM_OBJECT_ALLOCATION_FAILURE:
						fprintf( stderr, "RAN OUT OF MEMORY\n" );
						if ( !retried )
						{
							fprintf( stderr, "CLEARING MEMORY AND THEN RETRYING\n" );
							retried = 1;
							//gpuCache.InvalidateCache();
							//cpuQCache.InvalidateCache();
							myGpuCache->InvalidateCache();
							myCpuCache->InvalidateCache();
							goto wide_kernel_poly_opencl_retry_label;
						}
						else
						{
							fprintf( stderr, "ALREADY TRIED CLEARING MEMORY, GIVING UP\n" );
						}
						break;
					case CL_INVALID_EVENT_WAIT_LIST:
						fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
						break;
					case CL_OUT_OF_HOST_MEMORY:
						fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
						break;
					default:
						fprintf( stderr, "OTHER FAILURE: %i\n", errorCode );
						break;
				};
				return -1;
			}
			// debugging
			//fprintf( stdout, "Call to enqueue custom GEMV went well\n" );
			clFinish( kernelCommandQueue );
		}
		stopTimer();
		blasExecutionTime += calculateTime();
		// read the output
		// I really, sincerely don't care that gotos are bad practice
wide_kernel_poly_opencl_skip_work_label:
		startTimer();
		{
			if ( NULL != output )
			{
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, y_data, CL_TRUE, 0, sizeof(double) * numberOfJVectors, output, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR READING OUTPUT DATA: %i, %i, %i\n", i, startJ, endJ );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						case CL_OUT_OF_RESOURCES:
							fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
			}
		}
		stopTimer();
		outputReadingTime += calculateTime();
		
		// clean up
		{
			//clReleaseMemObject( x_data_i );
			//clReleaseMemObject( x_data_j );
			clReleaseMemObject( y_data );
		}
		return 0;
	}
	
	int wide_kernel_sigmoid_opencl( int i, int startJ, int endJ, double * output ) const
	{
		// variables
		clAmdBlasStatus blasStatus;
		cl_int errorCode;
		cl_mem x_data_i;
		cl_mem x_data_j;
		cl_mem y_data;
		cl_mem tempBuffer;
		//cl_mem q_i;
		uint32_t jDimension;
		uint32_t k;
		int numberOfJVectors;
		int qAlreadyComputed;
		int workDimension;
		int numberOfWorkGroups;
		int retried;
		double * tempData;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		profileDecls;
		GPUCache * myGpuCache;
		GPUCache * myCpuCache;
		GPUCache * myGpuQCache;
		
		// function body
		myGpuCache = (GPUCache*)(&gpuCache);
		myCpuCache = (GPUCache*)(&cpuQCache);
		myGpuQCache = (GPUCache*)(&gpuQCache);
		// see if we've already done the work
		{
			// calculate the number of vectors we need (just in case)
			numberOfJVectors = (endJ - startJ) + 1;
			qAlreadyComputed = 0;
			// all programming is an exercise in caching
			//if ( 0 == gpuQCache.CheckCache( i, &y_data ) )
			if ( 0 == myGpuQCache->CheckCache( i, &y_data ) )
			{
				qAlreadyComputed = 1;
				// I really, sincerely don't care that gotos are bad practice
				goto wide_kernel_sigmoid_opencl_skip_work_label;
			}
		}
		startTimer();
		retried = 0;
wide_kernel_sigmoid_opencl_retry_label:
		// write the ith vector out to the GPU
		{
			#ifdef _DENSE_REP
				//if ( -1 == gpuCache.CheckCache( i, &x_data_i ) )
				if ( -1 == myGpuCache->CheckCache( i, &x_data_i ) )
				{
					// debugging
					//fprintf( stdout, "MISSED CACHE ON X[%i]\n", i );
					x_data_i = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE/*ONLY*/, sizeof(double) * x[i].dim, NULL, &errorCode );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this
						fprintf( stderr, "ERROR CREATING X_DATA_I BUFFER\n" );
						return -1;
					}
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, x_data_i, CL_FALSE, 0, sizeof(double) * x[i].dim, x[i].values, 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this error
						fprintf( stderr, "ERROR WRITING X DATA TO BUFFER\n" );
						switch( errorCode )
						{
							case CL_MEM_OBJECT_ALLOCATION_FAILURE:
								fprintf( stderr, "RAN OUT OF MEMORY\n" );
								if ( !retried )
								{
									fprintf( stderr, "CLEARING MEMORY AND THEN RETRYING\n" );
									retried = 1;
									//gpuCache.InvalidateCache();
									//cpuQCache.InvalidateCache();
									myGpuCache->InvalidateCache();
									myCpuCache->InvalidateCache();
									goto wide_kernel_sigmoid_opencl_retry_label;
								}
								else
								{
									fprintf( stderr, "ALREADY TRIED CLEARING MEMORY, GIVING UP\n" );
								}
								break;
							case CL_INVALID_COMMAND_QUEUE:
								fprintf( stderr, "CL_INVALID_COMMAND_QUEUE\n" );
								break;
							case CL_INVALID_CONTEXT:
								fprintf( stderr, "CL_INVALID_CONTEXT\n" );
								break;
							case CL_INVALID_MEM_OBJECT:
								fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
								break;
							case CL_INVALID_VALUE:
								fprintf( stderr, "CL_INVALID_VALUE\n" );
								break;
							case CL_INVALID_EVENT_WAIT_LIST:
								fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
								break;
							case CL_OUT_OF_HOST_MEMORY:
								fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
								break;
							default:
								fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
								break;
						};
						return -1;
					}
					//gpuCache.CacheData( i, x_data_i );
					myGpuCache->CacheData( i, x_data_i );
				}
			#else
				// TODO: Handle nondense rep
			#endif
		}
		// write the other vectors out to the GPU
		{
			jDimension = x[startJ].dim;
			//if ( -1 == gpuCache.CheckCache( startJ, endJ, &x_data_j ) )
			if ( -1 == myGpuCache->CheckCache( startJ, endJ, &x_data_j ) )
			{
				// debugging
				//fprintf( stdout, "MISSED CACHE ON A\n" );
				// create the cl_mem buffer (assume all j vectors have same dimensionality)
				x_data_j = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfJVectors * jDimension, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING BUFFER FOR J VALUES\n" );
					return -1;
				}
				// loop through the j vectors and write them
				errorCode = 0;
				for ( k = 0; k < numberOfJVectors; k++ )
				{
					errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, x_data_j, CL_FALSE, k * sizeof(double) * jDimension, sizeof(double) * jDimension, x[k + startJ].values, 0, NULL, NULL );
				}
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR WRITING J VALUES TO GPU MEMORY\n" );
					return -1;
				}
				myGpuCache->CacheData( startJ, endJ, x_data_j );
				//gpuCache.CacheData( startJ, endJ, x_data_j );
			}
		}
		// wait for all the writing to finish
		{
			errorCode = clFinish( kernelCommandQueue );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this
				fprintf( stderr, "ERROR FINISHING WRITING EVENTS\n" );
				return -1;
			}
		}
		stopTimer();
		writingVectorsTime += calculateTime();
		startTimer();
		// set up the output
		{
			int remainder = numberOfJVectors % IDEAL_WORK_GROUP_SIZE;
			//if ( -1 == gpuCache.GetQSpace( numberOfJVectors, &y_data ) )
			//if( -1 == gpuQCache.CheckCache( i, &y_data ) )
			{
				y_data = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * (numberOfJVectors + IDEAL_WORK_GROUP_SIZE - remainder), NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING Y DATA BUFFER\n" );
					return -1;
				}
				//gpuQCache.CacheData( i, y_data );
			}
		}
		stopTimer();
		outputSetupTime += calculateTime();
		startTimer();
		// set up call to custom kernel
		{
			int remainder = jDimension % IDEAL_WORK_GROUP_SIZE;
			int workingSize = jDimension + IDEAL_WORK_GROUP_SIZE - remainder;
			numberOfWorkGroups = workingSize / IDEAL_WORK_GROUP_SIZE;
			int localSize = IDEAL_WORK_GROUP_SIZE;
			// set up work group dimensions
			workDimension = 2;
			globalWorkSize[0] = numberOfJVectors;
			globalWorkSize[1] = IDEAL_WORK_GROUP_SIZE; //workingSize;
			localWorkSize[0] = 1;
			localWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR CREATING TEMP BUFFER\n" );
				return -1;
			}
			// set up arguments
			// call will be of the form kernel( A, x, y, rows, cols, __local scratch, double gamma, double coef0, int degree )
			errorCode = clSetKernelArg( customMatrixVectorSigmoidKernel, 0, sizeof(cl_mem), &x_data_j );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 1, sizeof(cl_mem), &x_data_i );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 2, sizeof(cl_mem), &y_data );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 3, sizeof(int), &jDimension );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 4, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 5, sizeof(int), &localSize );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 6, sizeof(double), &gamma );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 7, sizeof(double), &coef0 );
			//errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 8, sizeof(int), &degree );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING KERNEL ARGS FOR CUSTOM GEMV\n" );
				return -1;
			}
		}
		// call our custom kernel (alternative to cl blas library)
		{
			// debugging
			//fprintf( stderr, "ABOUT TO RUN CUSTOM KERNEL\n" );
			// y = Ax
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, customMatrixVectorSigmoidKernel, 
												workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING CUSTOM MATRIX VECTOR KERNEL\n");
				switch( errorCode )
				{
					case CL_INVALID_PROGRAM_EXECUTABLE:
						fprintf( stderr, "CL_INVALID_PROGRAM_EXECUTABLE\n" );
						break;
					case CL_INVALID_COMMAND_QUEUE :
						fprintf( stderr, "CL_INVALID_COMMAND_QUEUE \n" );
						break;
					case CL_INVALID_KERNEL:
						fprintf( stderr, "CL_INVALID_KERNEL\n" );
						break;
					case CL_INVALID_CONTEXT:
						fprintf( stderr, "CL_INVALID_CONTEXT\n" );
						break;
					case CL_INVALID_KERNEL_ARGS:
						fprintf( stderr, "CL_INVALID_KERNEL_ARGS\n" );
						break;
					case CL_INVALID_WORK_DIMENSION:
						fprintf( stderr, "CL_INVALID_WORK_DIMENSION\n" );
						break;
					case CL_INVALID_WORK_GROUP_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_GROUP_SIZE: %i, %i\n", jDimension, CL_DEVICE_MAX_WORK_GROUP_SIZE );
						break;
					case CL_INVALID_WORK_ITEM_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_ITEM_SIZE\n" );
						break;
					case CL_INVALID_GLOBAL_OFFSET:
						fprintf( stderr, "CL_INVALID_GLOBAL_OFFSET\n" );
						break;
					case CL_OUT_OF_RESOURCES:
						fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
						break;
					case CL_MEM_OBJECT_ALLOCATION_FAILURE:
						fprintf( stderr, "RAN OUT OF MEMORY\n" );
						if ( !retried )
						{
							fprintf( stderr, "CLEARING MEMORY AND THEN RETRYING\n" );
							retried = 1;
							myGpuCache->InvalidateCache();//gpuCache.InvalidateCache();
							myCpuCache->InvalidateCache();//cpuQCache.InvalidateCache();
							goto wide_kernel_sigmoid_opencl_retry_label;
						}
						else
						{
							fprintf( stderr, "ALREADY TRIED CLEARING MEMORY, GIVING UP\n" );
						}
						break;
					case CL_INVALID_EVENT_WAIT_LIST:
						fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
						break;
					case CL_OUT_OF_HOST_MEMORY:
						fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
						break;
					default:
						fprintf( stderr, "OTHER FAILURE: %i\n", errorCode );
						break;
				};
				return -1;
			}
			// debugging
			//fprintf( stdout, "Call to enqueue custom GEMV went well\n" );
			clFinish( kernelCommandQueue );
		}
		stopTimer();
		blasExecutionTime += calculateTime();
		// read the output
		// I really, sincerely don't care that gotos are bad practice
wide_kernel_sigmoid_opencl_skip_work_label:
		startTimer();
		{
			if ( NULL != output )
			{
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, y_data, CL_TRUE, 0, sizeof(double) * numberOfJVectors, output, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR READING OUTPUT DATA: %i, %i, %i\n", i, startJ, endJ );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						case CL_OUT_OF_RESOURCES:
							fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
			}
		}
		stopTimer();
		outputReadingTime += calculateTime();
		
		// clean up
		{
			//clReleaseMemObject( x_data_i );
			//clReleaseMemObject( x_data_j );
			clReleaseMemObject( y_data );
		}
		return 0;
	}
	
	int wide_kernel_rbf_opencl( int i, int startJ, int endJ, double * output ) const
	{
	
		// variables
		clAmdBlasStatus blasStatus;
		cl_int errorCode;
		cl_mem x_data_i;
		cl_mem x_data_j;
		cl_mem y_data;
		cl_mem x_squareGpu;
		cl_mem tempBuffer;
		//cl_mem q_i;
		uint32_t jDimension;
		uint32_t k;
		int numberOfJVectors;
		int qAlreadyComputed;
		int workDimension;
		int numberOfWorkGroups;
		int retried;
		double * tempData;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		profileDecls;
		GPUCache * myGpuCache;
		GPUCache * myCpuCache;
		GPUCache * myGpuQCache;
		
		// function body
		myGpuCache = (GPUCache*)(&gpuCache);
		myCpuCache = (GPUCache*)(&cpuQCache);
		myGpuQCache = (GPUCache*)(&gpuQCache);
		// see if we've already done the work
		{
			// calculate the number of vectors we need (just in case)
			numberOfJVectors = (endJ - startJ) + 1;
			qAlreadyComputed = 0;
			// all programming is an exercise in caching
			//if ( 0 == gpuQCache.CheckCache( i, &y_data ) )
			if ( 0 == myGpuQCache->CheckCache( i, &y_data ) )
			{
				qAlreadyComputed = 1;
				// I really, sincerely don't care that gotos are bad practice
				goto wide_kernel_rbf_opencl_skip_work_label;
			}
		}
		startTimer();
		retried = 0;
wide_kernel_rbf_opencl_retry_label:
		// write the ith vector out to the GPU
		{
			#ifdef _DENSE_REP
				//if ( -1 == gpuCache.CheckCache( i, &x_data_i ) )
				if ( -1 == myGpuCache->CheckCache( i, &x_data_i ) )
				{
					// debugging
					//fprintf( stdout, "MISSED CACHE ON X[%i]\n", i );
					x_data_i = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE/*ONLY*/, sizeof(double) * x[i].dim, NULL, &errorCode );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this
						fprintf( stderr, "ERROR CREATING X_DATA_I BUFFER\n" );
						return -1;
					}
					errorCode = clEnqueueWriteBuffer( kernelCommandQueue, x_data_i, CL_FALSE, 0, sizeof(double) * x[i].dim, x[i].values, 0, NULL, NULL );
					if ( CL_SUCCESS != errorCode )
					{
						// TODO: Deal with this error
						fprintf( stderr, "ERROR WRITING X DATA TO BUFFER\n" );
						switch( errorCode )
						{
							case CL_MEM_OBJECT_ALLOCATION_FAILURE:
								fprintf( stderr, "RAN OUT OF MEMORY\n" );
								if ( !retried )
								{
									fprintf( stderr, "CLEARING MEMORY AND THEN RETRYING\n" );
									retried = 1;
									myGpuCache->InvalidateCache();//gpuCache.InvalidateCache();
									myCpuCache->InvalidateCache();//cpuQCache.InvalidateCache();
									goto wide_kernel_rbf_opencl_retry_label;
								}
								else
								{
									fprintf( stderr, "ALREADY TRIED CLEARING MEMORY, GIVING UP\n" );
								}
								break;
							case CL_INVALID_COMMAND_QUEUE:
								fprintf( stderr, "CL_INVALID_COMMAND_QUEUE\n" );
								break;
							case CL_INVALID_CONTEXT:
								fprintf( stderr, "CL_INVALID_CONTEXT\n" );
								break;
							case CL_INVALID_MEM_OBJECT:
								fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
								break;
							case CL_INVALID_VALUE:
								fprintf( stderr, "CL_INVALID_VALUE\n" );
								break;
							case CL_INVALID_EVENT_WAIT_LIST:
								fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
								break;
							case CL_OUT_OF_HOST_MEMORY:
								fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
								break;
							default:
								fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
								break;
						};
						return -1;
					}
					//gpuCache.CacheData( i, x_data_i );
					myGpuCache->CacheData( i, x_data_i );
				}
			#else
				// TODO: Handle nondense rep
			#endif
		}
		// write the other vectors out to the GPU
		{
			jDimension = x[startJ].dim;
			//if ( -1 == gpuCache.CheckCache( startJ, endJ, &x_data_j ) )
			if ( -1 == myGpuCache->CheckCache( startJ, endJ, &x_data_j ) )
			{
				// debugging
				//fprintf( stdout, "MISSED CACHE ON A\n" );
				// create the cl_mem buffer (assume all j vectors have same dimensionality)
				x_data_j = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfJVectors * jDimension, NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR CREATING BUFFER FOR J VALUES\n" );
					return -1;
				}
				// loop through the j vectors and write them
				errorCode = 0;
				for ( k = 0; k < numberOfJVectors; k++ )
				{
					errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, x_data_j, CL_FALSE, k * sizeof(double) * jDimension, sizeof(double) * jDimension, x[k + startJ].values, 0, NULL, NULL );
				}
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR WRITING J VALUES TO GPU MEMORY\n" );
					return -1;
				}
				myGpuCache->CacheData( startJ, endJ, x_data_j );
				//gpuCache.CacheData( startJ, endJ, x_data_j );
			}
		}
		// write the square vector
		{
			x_squareGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * numberOfVectors, NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR CREATING SPACE FOR X_SQUARE\n" );
				return -1;
			}
			errorCode = clEnqueueWriteBuffer( kernelCommandQueue, x_squareGpu, CL_FALSE, 0, sizeof(double) * numberOfVectors,
												x_square, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR WRITING X_SQUARE VECTOR\n" );
				return -1;
			}
		}
		// wait for all the writing to finish
		{
			errorCode = clFinish( kernelCommandQueue );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this
				fprintf( stderr, "ERROR FINISHING WRITING EVENTS\n" );
				return -1;
			}
		}
		stopTimer();
		writingVectorsTime += calculateTime();
		startTimer();
		// set up the output
		{
			int remainder = numberOfJVectors % IDEAL_WORK_GROUP_SIZE;
			//if ( -1 == gpuCache.GetQSpace( numberOfJVectors, &y_data ) )
			//if( -1 == gpuQCache.CheckCache( i, &y_data ) )
			{
				y_data = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * (numberOfJVectors + IDEAL_WORK_GROUP_SIZE - remainder), NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING Y DATA BUFFER\n" );
					return -1;
				}
				//gpuQCache.CacheData( i, y_data );
			}
		}
		stopTimer();
		outputSetupTime += calculateTime();
		startTimer();
		// set up call to custom kernel
		{
			int remainder = jDimension % IDEAL_WORK_GROUP_SIZE;
			int workingSize = jDimension + IDEAL_WORK_GROUP_SIZE - remainder;
			numberOfWorkGroups = workingSize / IDEAL_WORK_GROUP_SIZE;
			int localSize = IDEAL_WORK_GROUP_SIZE;
			// set up work group dimensions
			workDimension = 2;
			globalWorkSize[0] = numberOfJVectors;
			globalWorkSize[1] = IDEAL_WORK_GROUP_SIZE; //workingSize;
			localWorkSize[0] = 1;
			localWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR CREATING TEMP BUFFER\n" );
				return -1;
			}
			// set up arguments
			// call will be of the form kernel( A, x, y, rows, cols, __local scratch, double gamma, double coef0, int degree )
			errorCode = clSetKernelArg( customMatrixVectorRBFKernel, 0, sizeof(cl_mem), &x_data_j );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 1, sizeof(cl_mem), &x_data_i );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 2, sizeof(cl_mem), &y_data );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 3, sizeof(int), &jDimension );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 4, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 5, sizeof(int), &localSize );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 6, sizeof(double), &gamma );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 7, sizeof(double), &coef0 );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 8, sizeof(cl_mem), &x_squareGpu );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 9, sizeof(int), &i );
			//errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 8, sizeof(int), &degree );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING KERNEL ARGS FOR CUSTOM GEMV\n" );
				return -1;
			}
		}
		// call our custom kernel (alternative to cl blas library)
		{
			// debugging
			//fprintf( stderr, "ABOUT TO RUN CUSTOM KERNEL\n" );
			// y = Ax
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, customMatrixVectorRBFKernel, 
												workDimension, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING CUSTOM MATRIX VECTOR KERNEL\n");
				switch( errorCode )
				{
					case CL_INVALID_PROGRAM_EXECUTABLE:
						fprintf( stderr, "CL_INVALID_PROGRAM_EXECUTABLE\n" );
						break;
					case CL_INVALID_COMMAND_QUEUE :
						fprintf( stderr, "CL_INVALID_COMMAND_QUEUE \n" );
						break;
					case CL_INVALID_KERNEL:
						fprintf( stderr, "CL_INVALID_KERNEL\n" );
						break;
					case CL_INVALID_CONTEXT:
						fprintf( stderr, "CL_INVALID_CONTEXT\n" );
						break;
					case CL_INVALID_KERNEL_ARGS:
						fprintf( stderr, "CL_INVALID_KERNEL_ARGS\n" );
						break;
					case CL_INVALID_WORK_DIMENSION:
						fprintf( stderr, "CL_INVALID_WORK_DIMENSION\n" );
						break;
					case CL_INVALID_WORK_GROUP_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_GROUP_SIZE: %i, %i\n", jDimension, CL_DEVICE_MAX_WORK_GROUP_SIZE );
						break;
					case CL_INVALID_WORK_ITEM_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_ITEM_SIZE\n" );
						break;
					case CL_INVALID_GLOBAL_OFFSET:
						fprintf( stderr, "CL_INVALID_GLOBAL_OFFSET\n" );
						break;
					case CL_OUT_OF_RESOURCES:
						fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
						break;
					case CL_MEM_OBJECT_ALLOCATION_FAILURE:
						fprintf( stderr, "RAN OUT OF MEMORY\n" );
						if ( !retried )
						{
							fprintf( stderr, "CLEARING MEMORY AND THEN RETRYING\n" );
							retried = 1;
							myGpuCache->InvalidateCache();//gpuCache.InvalidateCache();
							myCpuCache->InvalidateCache();//cpuQCache.InvalidateCache();
							goto wide_kernel_rbf_opencl_retry_label;
						}
						else
						{
							fprintf( stderr, "ALREADY TRIED CLEARING MEMORY, GIVING UP\n" );
						}
						break;
					case CL_INVALID_EVENT_WAIT_LIST:
						fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
						break;
					case CL_OUT_OF_HOST_MEMORY:
						fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
						break;
					default:
						fprintf( stderr, "OTHER FAILURE: %i\n", errorCode );
						break;
				};
				return -1;
			}
			// debugging
			//fprintf( stdout, "Call to enqueue custom GEMV went well\n" );
			clFinish( kernelCommandQueue );
		}
		stopTimer();
		blasExecutionTime += calculateTime();
		// read the output
		// I really, sincerely don't care that gotos are bad practice
wide_kernel_rbf_opencl_skip_work_label:
		startTimer();
		{
			if ( NULL != output )
			{
				errorCode = clEnqueueReadBuffer( kernelCommandQueue, y_data, CL_TRUE, 0, sizeof(double) * numberOfJVectors, output, 0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR READING OUTPUT DATA: %i, %i, %i\n", i, startJ, endJ );
					switch( errorCode )
					{
						case CL_INVALID_COMMAND_QUEUE:
							fprintf( stderr, "INVALID COMMAND QUEUE\n" );
							break;
						case CL_INVALID_CONTEXT:
							fprintf( stderr, "CL_INVALID_CONTEXT\n" );
							break;
						case CL_INVALID_MEM_OBJECT:
							fprintf( stderr, "CL_INVALID_MEM_OBJECT\n" );
							break;
						case CL_INVALID_VALUE:
							fprintf( stderr, "CL_INVALID_VALUE\n" );
							break;
						case CL_INVALID_EVENT_WAIT_LIST:
							fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
							break;
						case CL_MEM_OBJECT_ALLOCATION_FAILURE:
							fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE\n" );
							break;
						case CL_OUT_OF_HOST_MEMORY:
							fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
							break;
						case CL_OUT_OF_RESOURCES:
							fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
							break;
						default:
							fprintf( stderr, "UNDOCUMENTED ERROR: %i\n", errorCode );
							break;
					};
					return -1;
				}
			}
		}
		stopTimer();
		outputReadingTime += calculateTime();
		
		// clean up
		{
			//clReleaseMemObject( x_data_i );
			//clReleaseMemObject( x_data_j );
			clReleaseMemObject( y_data );
			clReleaseMemObject( x_squareGpu );
		}
		return 0;
	}
	
	// set of helper functions for prediction
	double prediction_setup_linear( struct svm_node * xData, double * svmCoefficients )
	{
	
		// variables
		cl_int errorCode;
		cl_mem A;
		cl_mem alpha;
		cl_mem xDataGpu;
		cl_mem yDataGpu;
		cl_mem sumGpu;
		int k;
		double * yDataCpu;
		double sum;
		
		// function body
		// write x data to the GPU
		{
			xDataGpu = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * xData->dim, NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR VECTOR TO BE PREDICTED\n" );
				exit( -1 );
			}
			errorCode = clEnqueueWriteBuffer( kernelCommandQueue, xDataGpu, CL_FALSE, 0, 
												sizeof(double) * (xData->dim), xData->values,
												0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR WRITING X DATA TO GPU\n" );
				exit( -1 );
			}
		}
		// set up A matrix
		{
			if ( 0 != gpuCache.CheckCache( 0, numberOfVectors-1, &A ) )
			{
				// debugging
				fprintf( stdout, "WRITING A\n" );
				A = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfVectors * x[0].dim,
									NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR CREATING SPACE FOR A DURING PREDICTION\n" );
					exit( -1 );
				}
				// loop through the j vectors and write them
				errorCode = 0;
				for ( k = 0; k < numberOfVectors; k++ )
				{
					errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, A, CL_FALSE, 
														k * sizeof(double) * x[0].dim, 
														sizeof(double) * x[0].dim, x[k].values, 0, NULL, NULL );
				}
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR WRITING A VALUES TO GPU MEMORY\n" );
					return -1;
				}
				gpuCache.CacheData( 0, numberOfVectors-1, A );
			}
		}
		// write SV coefficients to GPU
		{
			// we're gonna cheat, using the knowledge that individual x values don't get cached during classification
			if ( 0 != gpuCache.CheckCache( 0, &alpha ) )
			{ 
				// debugging
				fprintf( stdout, "WRITING ALPHA\n" );
				alpha = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfVectors,
										NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING BUFFER TO HOLD SV COEFFICIENTS\n" );
					exit( -1 );
				}
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, alpha, CL_FALSE, 0,
													sizeof(double) * numberOfVectors, svmCoefficients,
													0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING SV COEFFICIENTS TO GPU\n" );
					exit( -1 );
				}
				gpuCache.CacheData( 0, alpha );
			}
		}
		// set up intermediate and output data structures
		{
			yDataGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * numberOfVectors,
										NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR Y VECTOR DURING PREDICTION\n" );
				exit( -1 );
			}
			sumGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double), NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR SUM DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// wait for writes to finish
		{
			clFinish( kernelCommandQueue );
		}
		// make call to linear kernel
		{
			if ( 0 != run_linear_kernel( xDataGpu, A, yDataGpu ) )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING LINEAR KERNER DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// clear the queue
		{
			if ( CL_SUCCESS != ( errorCode = clFinish( kernelCommandQueue ) ) )
			{
				fprintf( stderr, "ERROR FINISHING QUEUE\n" );
				exit( -1 );
			}
		}
		// make call to reduction kernel
		{
			if ( 0 != run_prediction_reduction_kernel( yDataGpu, alpha, sumGpu ) )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING REDUCTION KERNEL DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// clear the queue
		{
			clFinish( kernelCommandQueue );
		}
		// read back output
		{
			errorCode = clEnqueueReadBuffer( kernelCommandQueue, sumGpu, CL_TRUE, 0, sizeof(double),
											 &sum, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR READING SUM BACK FROM GPU\n" );
				exit( -1 );
			}
		}
		
		// clean up
		{
			clReleaseMemObject( xDataGpu );
			clReleaseMemObject( yDataGpu );
			clReleaseMemObject( sumGpu );
			//clReleaseMemObject( alpha );
		}
		return sum;
	}
	
	int run_linear_kernel( cl_mem & xData, cl_mem & aData, cl_mem & yData )
	{
		// variables
		cl_int errorCode;
		int workDimension;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		
		// function body
		// set up dimensions for linear kernel
		{
			// it's a two phase reduction algorithm
			workDimension = 2;
			globalWorkSize[0] = numberOfVectors;
			globalWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
			localWorkSize[0] = 1;
			localWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
		}
		// set up arguments for linear kernel
		// call is of the form: kernel( A, x, y, cols, __local scratch, localSize )
		{
			errorCode = clSetKernelArg( customMatrixVectorKernel, 0, sizeof(cl_mem), &aData );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 1, sizeof(cl_mem), &xData );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 2, sizeof(cl_mem), &yData );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 3, sizeof(int), &(x[0].dim) );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 4, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			errorCode |= clSetKernelArg( customMatrixVectorKernel, 5, sizeof(int), &(localWorkSize[1]) );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING ARGUMENTS FOR CALL TO LINEAR KERNEL\n" );
				return -1;
			}
		}
		// call the linear kernel
		{
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, customMatrixVectorKernel, workDimension,
												NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CALLING LINEAR KERNEL\n" );
				switch( errorCode )
				{
					case CL_INVALID_PROGRAM_EXECUTABLE:
						fprintf( stderr, "CL_INVALID_PROGRAM_EXECUTABLE\n" );
						break;
					case CL_INVALID_COMMAND_QUEUE :
						fprintf( stderr, "CL_INVALID_COMMAND_QUEUE \n" );
						break;
					case CL_INVALID_KERNEL:
						fprintf( stderr, "CL_INVALID_KERNEL\n" );
						break;
					case CL_INVALID_CONTEXT:
						fprintf( stderr, "CL_INVALID_CONTEXT\n" );
						break;
					case CL_INVALID_KERNEL_ARGS:
						fprintf( stderr, "CL_INVALID_KERNEL_ARGS\n" );
						break;
					case CL_INVALID_WORK_DIMENSION:
						fprintf( stderr, "CL_INVALID_WORK_DIMENSION\n" );
						break;
					case CL_INVALID_WORK_GROUP_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_GROUP_SIZE\n" );
						break;
					case CL_INVALID_WORK_ITEM_SIZE:
						fprintf( stderr, "CL_INVALID_WORK_ITEM_SIZE\n" );
						break;
					case CL_INVALID_GLOBAL_OFFSET:
						fprintf( stderr, "CL_INVALID_GLOBAL_OFFSET\n" );
						break;
					case CL_OUT_OF_RESOURCES:
						fprintf( stderr, "CL_OUT_OF_RESOURCES\n" );
						break;
					case CL_MEM_OBJECT_ALLOCATION_FAILURE:
						fprintf( stderr, "RAN OUT OF MEMORY\n" );
						break;
					case CL_INVALID_EVENT_WAIT_LIST:
						fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST\n" );
						break;
					case CL_OUT_OF_HOST_MEMORY:
						fprintf( stderr, "CL_OUT_OF_HOST_MEMORY\n" );
						break;
					default:
						fprintf( stderr, "OTHER FAILURE: %i\n", errorCode );
						break;
				};
				return -1;
			}
		}
		
		// clean up
		return 0;
	}
	
	double prediction_setup_poly( struct svm_node * xData, double * svmCoefficients )
	{
	
		// variables
		cl_int errorCode;
		cl_mem A;
		cl_mem alpha;
		cl_mem xDataGpu;
		cl_mem yDataGpu;
		cl_mem sumGpu;
		int k;
		double * yDataCpu;
		double sum;
		
		// function body
		// write x data to the GPU
		{
			xDataGpu = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * xData->dim, NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR VECTOR TO BE PREDICTED\n" );
				exit( -1 );
			}
			errorCode = clEnqueueWriteBuffer( kernelCommandQueue, xDataGpu, CL_FALSE, 0, 
												sizeof(double) * (xData->dim), xData->values,
												0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR WRITING X DATA TO GPU\n" );
				exit( -1 );
			}
		}
		// set up A matrix
		{
			if ( 0 != gpuCache.CheckCache( 0, numberOfVectors-1, &A ) )
			{
				A = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfVectors * x[0].dim,
									NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR CREATING SPACE FOR A DURING PREDICTION\n" );
					exit( -1 );
				}
				// loop through the j vectors and write them
				errorCode = 0;
				for ( k = 0; k < numberOfVectors; k++ )
				{
					errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, A, CL_FALSE, 
														k * sizeof(double) * x[0].dim, 
														sizeof(double) * x[0].dim, x[k].values, 0, NULL, NULL );
				}
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR WRITING A VALUES TO GPU MEMORY\n" );
					return -1;
				}
				gpuCache.CacheData( 0, numberOfVectors-1, A );
			}
		}
		// write SV coefficients to GPU
		{
			// we're gonna cheat, using the knowledge that individual x values don't get cached during classification
			if ( 0 != gpuCache.CheckCache( 0, &alpha ) )
			{
				alpha = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfVectors,
										NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING BUFFER TO HOLD SV COEFFICIENTS\n" );
					exit( -1 );
				}
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, alpha, CL_FALSE, 0,
													sizeof(double) * numberOfVectors, svmCoefficients,
													0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING SV COEFFICIENTS TO GPU\n" );
					exit( -1 );
				}
				gpuCache.CacheData( 0, alpha );
			}
		}
		// set up intermediate and output data structures
		{
			yDataGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * numberOfVectors,
										NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR Y VECTOR DURING PREDICTION\n" );
				exit( -1 );
			}
			sumGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double), NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR SUM DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// wait for writes to finish
		{
			clFinish( kernelCommandQueue );
		}
		// make call to linear kernel
		{
			if ( 0 != run_poly_kernel( xDataGpu, A, yDataGpu ) )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING LINEAR KERNER DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// clear the queue
		{
			if ( CL_SUCCESS != ( errorCode = clFinish( kernelCommandQueue ) ) )
			{
				fprintf( stderr, "ERROR FINISHING QUEUE\n" );
				exit( -1 );
			}
		}
		// make call to reduction kernel
		{
			if ( 0 != run_prediction_reduction_kernel( yDataGpu, alpha, sumGpu ) )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING REDUCTION KERNEL DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// clear the queue
		{
			clFinish( kernelCommandQueue );
		}
		// read back output
		{
			errorCode = clEnqueueReadBuffer( kernelCommandQueue, sumGpu, CL_TRUE, 0, sizeof(double),
											 &sum, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR READING SUM BACK FROM GPU\n" );
				exit( -1 );
			}
		}
		
		// clean up
		{
			clReleaseMemObject( xDataGpu );
			clReleaseMemObject( sumGpu );
			clReleaseMemObject( yDataGpu );
			//clReleaseMemObject( alpha );
		}
		return sum;
	}

	int run_poly_kernel( cl_mem & xData, cl_mem & aData, cl_mem & yData )
	{
		// variables
		cl_int errorCode;
		int workDimension;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		
		// function body
		// set up dimensions for linear kernel
		{
			// it's a two phase reduction algorithm
			workDimension = 2;
			globalWorkSize[0] = numberOfVectors;
			globalWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
			localWorkSize[0] = 1;
			localWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
		}
		// set up arguments for linear kernel
		// call is of the form: kernel( A, x, y, cols, __local scratch, localSize, gamma, coef0, degree )
		{
			errorCode = clSetKernelArg( customMatrixVectorPolynomialKernel, 0, sizeof(cl_mem), &aData );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 1, sizeof(cl_mem), &xData );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 2, sizeof(cl_mem), &yData );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 3, sizeof(int), &(x[0].dim) );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 4, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 5, sizeof(int), &(localWorkSize[1]) );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 6, sizeof(double), &gamma );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 7, sizeof(double), &coef0 );
			errorCode |= clSetKernelArg( customMatrixVectorPolynomialKernel, 8, sizeof(int), &degree );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING ARGUMENTS FOR CALL TO LINEAR KERNEL\n" );
				return -1;
			}
		}
		// call the linear kernel
		{
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, customMatrixVectorPolynomialKernel, workDimension,
												NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CALLING LINEAR KERNEL\n" );
				return -1;
			}
		}
		
		// clean up
		return 0;
	}

	double prediction_setup_sigmoid( struct svm_node * xData, double * svmCoefficients )
	{
	
		// variables
		cl_int errorCode;
		cl_mem A;
		cl_mem alpha;
		cl_mem xDataGpu;
		cl_mem yDataGpu;
		cl_mem sumGpu;
		int k;
		double * yDataCpu;
		double sum;
		
		// function body
		// write x data to the GPU
		{
			xDataGpu = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * xData->dim, NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR VECTOR TO BE PREDICTED\n" );
				exit( -1 );
			}
			errorCode = clEnqueueWriteBuffer( kernelCommandQueue, xDataGpu, CL_FALSE, 0, 
												sizeof(double) * (xData->dim), xData->values,
												0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR WRITING X DATA TO GPU\n" );
				exit( -1 );
			}
		}
		// set up A matrix
		{
			if ( 0 != gpuCache.CheckCache( 0, numberOfVectors-1, &A ) )
			{
				A = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfVectors * x[0].dim,
									NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR CREATING SPACE FOR A DURING PREDICTION\n" );
					exit( -1 );
				}
				// loop through the j vectors and write them
				errorCode = 0;
				for ( k = 0; k < numberOfVectors; k++ )
				{
					errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, A, CL_FALSE, 
														k * sizeof(double) * x[0].dim, 
														sizeof(double) * x[0].dim, x[k].values, 0, NULL, NULL );
				}
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR WRITING A VALUES TO GPU MEMORY\n" );
					return -1;
				}
				gpuCache.CacheData( 0, numberOfVectors-1, A );
			}
		}
		// write SV coefficients to GPU
		{
			// we're gonna cheat, using the knowledge that individual x values don't get cached during classification
			if ( 0 != gpuCache.CheckCache( 0, &alpha ) )
			{
				alpha = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfVectors,
										NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING BUFFER TO HOLD SV COEFFICIENTS\n" );
					exit( -1 );
				}
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, alpha, CL_FALSE, 0,
													sizeof(double) * numberOfVectors, svmCoefficients,
													0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING SV COEFFICIENTS TO GPU\n" );
					exit( -1 );
				}
				gpuCache.CacheData( 0, alpha );
			}
		}
		// set up intermediate and output data structures
		{
			yDataGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * numberOfVectors,
										NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR Y VECTOR DURING PREDICTION\n" );
				exit( -1 );
			}
			sumGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double), NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR SUM DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// wait for writes to finish
		{
			clFinish( kernelCommandQueue );
		}
		// make call to sigmoid kernel
		{
			if ( 0 != run_sigmoid_kernel( xDataGpu, A, yDataGpu ) )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING LINEAR KERNER DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// clear the queue
		{
			if ( CL_SUCCESS != ( errorCode = clFinish( kernelCommandQueue ) ) )
			{
				fprintf( stderr, "ERROR FINISHING QUEUE\n" );
				exit( -1 );
			}
		}
		// make call to reduction kernel
		{
			if ( 0 != run_prediction_reduction_kernel( yDataGpu, alpha, sumGpu ) )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING REDUCTION KERNEL DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// clear the queue
		{
			clFinish( kernelCommandQueue );
		}
		// read back output
		{
			errorCode = clEnqueueReadBuffer( kernelCommandQueue, sumGpu, CL_TRUE, 0, sizeof(double),
											 &sum, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR READING SUM BACK FROM GPU\n" );
				exit( -1 );
			}
		}
		
		// clean up
		{
			clReleaseMemObject( xDataGpu );
			clReleaseMemObject( sumGpu );
			clReleaseMemObject( yDataGpu );
			//clReleaseMemObject( alpha );
		}
		return sum;
	}

	int run_sigmoid_kernel( cl_mem & xData, cl_mem & aData, cl_mem & yData )
	{
		// variables
		cl_int errorCode;
		int workDimension;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		
		// function body
		// set up dimensions for linear kernel
		{
			// it's a two phase reduction algorithm
			workDimension = 2;
			globalWorkSize[0] = numberOfVectors;
			globalWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
			localWorkSize[0] = 1;
			localWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
		}
		// set up arguments for linear kernel
		// call is of the form: kernel( A, x, y, cols, __local scratch, localSize, gamma, coef0)
		{
			errorCode = clSetKernelArg( customMatrixVectorSigmoidKernel, 0, sizeof(cl_mem), &aData );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 1, sizeof(cl_mem), &xData );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 2, sizeof(cl_mem), &yData );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 3, sizeof(int), &(x[0].dim) );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 4, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 5, sizeof(int), &(localWorkSize[1]) );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 6, sizeof(double), &gamma );
			errorCode |= clSetKernelArg( customMatrixVectorSigmoidKernel, 7, sizeof(double), &coef0 );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING ARGUMENTS FOR CALL TO LINEAR KERNEL\n" );
				return -1;
			}
		}
		// call the linear kernel
		{
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, customMatrixVectorSigmoidKernel, workDimension,
												NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CALLING LINEAR KERNEL\n" );
				return -1;
			}
		}
		
		// clean up
		return 0;
	}

	double prediction_setup_rbf( struct svm_node * xData, double * svmCoefficients )
	{
	
		// variables
		cl_int errorCode;
		cl_mem A;
		cl_mem alpha;
		cl_mem xDataGpu;
		cl_mem yDataGpu;
		cl_mem sumGpu;
		int k;
		double * yDataCpu;
		double sum;
		
		// function body
		// write x data to the GPU
		{
			xDataGpu = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * xData->dim, NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR VECTOR TO BE PREDICTED\n" );
				exit( -1 );
			}
			errorCode = clEnqueueWriteBuffer( kernelCommandQueue, xDataGpu, CL_FALSE, 0, 
												sizeof(double) * (xData->dim), xData->values,
												0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR WRITING X DATA TO GPU\n" );
				exit( -1 );
			}
		}
		// set up A matrix
		{
			if ( 0 != gpuCache.CheckCache( 0, numberOfVectors-1, &A ) )
			{
				A = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfVectors * x[0].dim,
									NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					fprintf( stderr, "ERROR CREATING SPACE FOR A DURING PREDICTION\n" );
					exit( -1 );
				}
				// loop through the j vectors and write them
				errorCode = 0;
				for ( k = 0; k < numberOfVectors; k++ )
				{
					errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, A, CL_FALSE, 
														k * sizeof(double) * x[0].dim, 
														sizeof(double) * x[0].dim, x[k].values, 0, NULL, NULL );
				}
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this
					fprintf( stderr, "ERROR WRITING A VALUES TO GPU MEMORY\n" );
					return -1;
				}
				gpuCache.CacheData( 0, numberOfVectors-1, A );
			}
		}
		// write SV coefficients to GPU
		{
			// we're gonna cheat, using the knowledge that individual x values don't get cached during classification
			if ( 0 != gpuCache.CheckCache( 0, &alpha ) )
			{
				alpha = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * numberOfVectors,
										NULL, &errorCode );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR CREATING BUFFER TO HOLD SV COEFFICIENTS\n" );
					exit( -1 );
				}
				errorCode = clEnqueueWriteBuffer( kernelCommandQueue, alpha, CL_FALSE, 0,
													sizeof(double) * numberOfVectors, svmCoefficients,
													0, NULL, NULL );
				if ( CL_SUCCESS != errorCode )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR WRITING SV COEFFICIENTS TO GPU\n" );
					exit( -1 );
				}
				gpuCache.CacheData( 0, alpha );
			}
		}
		// set up intermediate and output data structures
		{
			yDataGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double) * numberOfVectors,
										NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR Y VECTOR DURING PREDICTION\n" );
				exit( -1 );
			}
			sumGpu = clCreateBuffer( kernelContext, CL_MEM_READ_WRITE, sizeof(double), NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CREATING SPACE FOR SUM DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// wait for writes to finish
		{
			clFinish( kernelCommandQueue );
		}
		// make call to rbf kernel
		{
			if ( 0 != run_rbf_kernel( xDataGpu, A, yDataGpu, xData ) )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING LINEAR KERNER DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// clear the queue
		{
			if ( CL_SUCCESS != ( errorCode = clFinish( kernelCommandQueue ) ) )
			{
				fprintf( stderr, "ERROR FINISHING QUEUE\n" );
				exit( -1 );
			}
		}
		// make call to reduction kernel
		{
			if ( 0 != run_prediction_reduction_kernel( yDataGpu, alpha, sumGpu ) )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR RUNNING REDUCTION KERNEL DURING PREDICTION\n" );
				exit( -1 );
			}
		}
		// clear the queue
		{
			clFinish( kernelCommandQueue );
		}
		// read back output
		{
			errorCode = clEnqueueReadBuffer( kernelCommandQueue, sumGpu, CL_TRUE, 0, sizeof(double),
											 &sum, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR READING SUM BACK FROM GPU\n" );
				exit( -1 );
			}
		}
		
		// clean up
		{
			clReleaseMemObject( xDataGpu );
			clReleaseMemObject( sumGpu );
			clReleaseMemObject( yDataGpu );
			//clReleaseMemObject( alpha );
		}
		return sum;
	}

	int run_rbf_kernel( cl_mem & xData, cl_mem & aData, cl_mem & yData, struct svm_node * xDataCpu )
	{
		// variables
		cl_int errorCode;
		cl_mem xSquareGpu;
		double * tempXSquare;
		int workDimension;
		int k;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		
		// function body
		// set up dimensions for linear kernel
		{
			// it's a two phase reduction algorithm
			workDimension = 2;
			globalWorkSize[0] = numberOfVectors;
			globalWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
			localWorkSize[0] = 1;
			localWorkSize[1] = IDEAL_WORK_GROUP_SIZE;
		}
		// prepare a modified xSquare
		// TODO: Cache this (?)
		{
			// allocate space for all the square values + 1
			tempXSquare = (double*) malloc( sizeof(double) * (numberOfVectors + 1) );
			if ( NULL != tempXSquare )
			{
				fprintf( stderr, "RAN OUT OF HOST MEMORY IN RUN_RBF_KERNEL\n" );
				return -1;
			}
			// copy over xSquare
			for ( k = 0; k < numberOfVectors; k++ )
			{
				tempXSquare[ k ] = x_square[ k ];
			}
			// set up the last one
			tempXSquare[ numberOfVectors ] = dot( xDataCpu, xDataCpu );
			// write it to the GPU
			xSquareGpu = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * (numberOfVectors + 1), NULL, &errorCode );
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR CREATING XSQUARE GPU VALUE\n" );
				return -1;
			}
			errorCode = clEnqueueWriteBuffer( kernelCommandQueue, xSquareGpu, CL_TRUE, 0, sizeof(double) * (numberOfVectors + 1), tempXSquare, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR WRITING XSQUARE DATA TO GPU\n" );
				return -1;
			}
		}
		// set up arguments for linear kernel
		// call is of the form: kernel( A, x, y, cols, __local scratch, localSize, gamma, coef0, xSquare, i )
		{
			errorCode = clSetKernelArg( customMatrixVectorRBFKernel, 0, sizeof(cl_mem), &aData );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 1, sizeof(cl_mem), &xData );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 2, sizeof(cl_mem), &yData );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 3, sizeof(int), &(x[0].dim) );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 4, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 5, sizeof(int), &(localWorkSize[1]) );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 6, sizeof(double), &gamma );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 7, sizeof(double), &coef0 );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 8, sizeof(cl_mem), &xSquareGpu );
			errorCode |= clSetKernelArg( customMatrixVectorRBFKernel, 9, sizeof(int), &numberOfVectors );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING ARGUMENTS FOR CALL TO LINEAR KERNEL\n" );
				return -1;
			}
		}
		// call the linear kernel
		{
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, customMatrixVectorRBFKernel, workDimension,
												NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR CALLING LINEAR KERNEL\n" );
				return -1;
			}
		}
		
		// clean up
		return 0;
	}

	int run_prediction_reduction_kernel( cl_mem & y, cl_mem & alpha, cl_mem & sum )
	{
		// variables
		cl_int errorCode;
		int workDimension;
		size_t globalWorkSize[3];
		size_t localWorkSize[3];
		
		// function body
		// set work group dimensions for prediction reduction kernel
		{
			workDimension = 1;
			globalWorkSize[0] = IDEAL_WORK_GROUP_SIZE;
			localWorkSize[0] = IDEAL_WORK_GROUP_SIZE;
		}
		// set arguments for prediction reduction kernel
		{
			// call will be of the form: kernel( y, alpha, __local scratch, sum, cols, workGroupSize )
			errorCode = clSetKernelArg( predictionReductionKernel, 0, sizeof(cl_mem), &y );
			errorCode |= clSetKernelArg( predictionReductionKernel, 1, sizeof(cl_mem), &alpha );
			errorCode |= clSetKernelArg( predictionReductionKernel, 2, sizeof(double) * IDEAL_WORK_GROUP_SIZE, NULL );
			errorCode |= clSetKernelArg( predictionReductionKernel, 3, sizeof(cl_mem), &sum );
			errorCode |= clSetKernelArg( predictionReductionKernel, 4, sizeof(int), &numberOfVectors );
			errorCode |= clSetKernelArg( predictionReductionKernel, 5, sizeof(int), &(localWorkSize[0]) );
			if ( CL_SUCCESS != errorCode )
			{
				// TODO: Deal with this error
				fprintf( stderr, "ERROR SETTING KERNEL ARGS\n" );
				return -1;
			}
		}
		// run prediction reduction kernel
		{
			errorCode = clEnqueueNDRangeKernel( kernelCommandQueue, predictionReductionKernel, workDimension,
												NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
			if ( CL_SUCCESS != errorCode )
			{
				fprintf( stderr, "ERROR ENQUEUING PREDICTION REDUCTION KERNEL\n" );
				return -1;
			}
		}
		
		// clean up
		return 0;
	}

	#endif
	
};

#define	GPU_CACHE_SIZE	6000
#define	GPU_Q_CACHE_SIZE	6000
#define	CPU_Q_CACHE_SIZE	6000

#ifdef _DENSE_REP
Kernel::Kernel(int l, svm_node * x_, const svm_parameter& param)
#else
Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
#endif
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0), gpuCache( l, GPU_CACHE_SIZE ), gpuQCache( l,GPU_Q_CACHE_SIZE ),
 cpuQCache( l, CPU_Q_CACHE_SIZE )
{


	#ifdef	CL_SVM
	
	linearKernelKernelSource = 
	#include "linearKernelKernelSource.cl"
	;
	customDaxpyKernelSource = 
	#include "customDaxpyKernelSource.cl"
	;
	swapObjectiveFunctionKernelSource = 
	#include "swapObjectiveFunctionKernelSource.cl"
	;
	findCandidateIValuesKernelSource = 
	#include "findCandidateIValuesKernelSource.cl"
	;
	findCandidateJValuesKernelSource = 
	#include "findCandidateJValuesKernelSource.cl"
	;
	dualDaxpyKernelSource =
	#include "dualDaxpyKernelSource.cl"
	;
	customMatrixVectorKernelSource = 
	#include "customMatrixVectorKernelSource.cl"
	;
	 reductionKernelSource = 
	#include "reductionKernelSource.cl"
	;
	 swapVectorBlockKernelSource =
	#include "swapVectorBlockKernelSource.cl"
	;
	 customMatrixVectorPolynomialKernelSource =
	#include "customMatrixVectorPolynomialKernelSource.cl"
	;
	 customMatrixVectorSigmoidKernelSource =
	#include "customMatrixVectorSigmoidKernelSource.cl"
	;
	 customMatrixVectorRBFKernelSource =
	#include "customMatrixVectorRBFKernelSource.cl"
	;
	 predictionReductionKernelSource = 
	#include "predictionReductionKernelSource.cl"
	;
	
		// profiling, TEMPORARY
		otherFunctionTime = 0;
		gettingQTime = 0;
		gettingGTime = 0;
		daxpyTime = 0;
		swappingTime = 0;
		swappingCount = 0;
		qRetrievingTime = 0;
		swappingOtherFunctionTime = 0;
		qCacheSwappingTime = 0;
		endQueueTime = 0;
		kernelEnqueuingTime = 0;
		argSettingTime = 0;
	
	#define CL_PLATFORM_COUNT	10
		// variables
		cl_device_id firstDevice;
		cl_device_id cpuDevice;
		//cl_platform_id firstPlatform;
		cl_platform_id platformIDs[CL_PLATFORM_COUNT];
		cl_program linearKernelKernelProgram;
		cl_program customDaxpyKernelProgram;
		cl_program swapObjectiveFunctionKernelProgram;
		cl_program findCandidateIValuesKernelProgram;
		cl_program findCandidateJValuesKernelProgram;
		cl_program dualDaxpyKernelProgram;
		cl_program customMatrixVectorKernelProgram;
		cl_program customMatrixVectorPolynomialKernelProgram;
		cl_program customMatrixVectorSigmoidKernelProgram;
		cl_program customMatrixVectorRBFKernelProgram;
		cl_program reductionKernelProgram;
		cl_program swapVectorBlockKernelProgram;
		cl_program predictionReductionKernelProgram;
		cl_uint numberOfEntries;
		cl_int errorCode;
		uint32_t k;
		
		// get the device id
		if ( 0 != clGetPlatformIDs( CL_PLATFORM_COUNT, platformIDs/*&firstPlatform*/, &numberOfEntries ) ||
			0 == numberOfEntries )
		{
			// TODO:
			// replace this placeholder
			// debugging
			fprintf( stderr, "Error getting platform IDs\n" );
			exit( -1 );
		}
		// debugging
		fprintf( stdout, "Number of OpenCL platforms: %i\n", numberOfEntries );
		// debugging
		fprintf( stdout, "Scanning for GPU devices\n" );
		k = 0;
		do
		{
			errorCode = clGetDeviceIDs( platformIDs[k]/*firstPlatform*/, 
						CL_DEVICE_TYPE_GPU,
						1,
						&firstDevice,
						&numberOfEntries );
			if ( ( CL_SUCCESS != errorCode && CL_DEVICE_NOT_FOUND != errorCode ) )//|| 0 == numberOfEntries )
			{
				// TODO:
				// replace this placeholder
				// debugging
				fprintf( stderr, "Error getting device IDs\n" );
				switch( errorCode )
				{
					case CL_INVALID_PLATFORM:
						fprintf( stderr, "INVALID PLATFORM\n" );
						break;
					case CL_INVALID_DEVICE_TYPE:
						fprintf( stderr, "CL_INVALID_DEVICE_TYPE\n" );
						break;
					case CL_INVALID_VALUE:
						fprintf( stderr, "CL_INVALID_VALUE\n" );
						break;
					case CL_DEVICE_NOT_FOUND:
						fprintf( stderr, "CL_DEVICE_NOT_FOUND\n" );
						break;
				};
				exit( -1 );
			}
			k++;
		} while ( CL_DEVICE_NOT_FOUND == errorCode && k < CL_PLATFORM_COUNT );
		// debugging
		fprintf( stdout, "Located GPU Device\n" );
	
		fprintf( stdout, "Scanning for CPU devices\n" );
		k = 0;
		do
		{
			errorCode = clGetDeviceIDs( platformIDs[k]/*firstPlatform*/, 
						CL_DEVICE_TYPE_CPU,
						1,
						&cpuDevice,
						&numberOfEntries );
			if ( ( CL_SUCCESS != errorCode && CL_DEVICE_NOT_FOUND != errorCode ) )//|| 0 == numberOfEntries )
			{
				// TODO:
				// replace this placeholder
				// debugging
				fprintf( stderr, "Error getting device IDs\n" );
				switch( errorCode )
				{
					case CL_INVALID_PLATFORM:
						fprintf( stderr, "INVALID PLATFORM\n" );
						break;
					case CL_INVALID_DEVICE_TYPE:
						fprintf( stderr, "CL_INVALID_DEVICE_TYPE\n" );
						break;
					case CL_INVALID_VALUE:
						fprintf( stderr, "CL_INVALID_VALUE\n" );
						break;
					case CL_DEVICE_NOT_FOUND:
						fprintf( stderr, "CL_DEVICE_NOT_FOUND\n" );
						break;
				};
				exit( -1 );
			}
			k++;
		} while ( CL_DEVICE_NOT_FOUND == errorCode && k < CL_PLATFORM_COUNT );
		// debugging
		fprintf( stdout, "Located CPU Device\n" );
		
	
		// create a compute context
		kernelContext = clCreateContext( 0, 1, &firstDevice,
							/** pfn_notify/*/NULL/**/, NULL, &errorCode );
		if ( 0 != errorCode )
		{
			// TODO:
			// replace this placeholder
			// debugging
			fprintf( stderr, "Error creating context\n" );
			exit( -1 );
		}
		// create the command queue
		kernelCommandQueue = clCreateCommandQueue( kernelContext,
								  firstDevice,
								  0,//CL_QUEUE_PROFILING_ENABLE,
								  &errorCode );
		if ( 0 != errorCode )
		{
			// TODO:
			// replace this placeholder
			// debugging
			fprintf( stderr, "Error creating command queue\n" );
			exit( -1 );
		}

		// create a CPU based compute context
		/*
		kernelCpuContext = clCreateContext( 0, 1, &cpuDevice,
											NULL, NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR CREATING CPU KERNEL CONTEXT\n" );
			exit( -1 );
		}
		/*/
		kernelCpuContext = kernelContext;
		//*/
		/*
		kernelCpuCommandQueue = clCreateCommandQueue( kernelCpuContext,
														cpuDevice,
														0,
														&errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR CREATING CPU KERNEL COMMAND QUEUE\n" );
			exit( -1 );
		}
		/*/
		kernelCpuCommandQueue = kernelCommandQueue;
		cpuDevice = firstDevice;
		//*/
		
		// start whipping up kernels

		linearKernelKernelProgram = clCreateProgramWithSource( kernelContext,
								1,
								&linearKernelKernelSource,
								NULL,
								&errorCode );
		if ( 0 != errorCode )
		{
			// TODO:
			// replace this placeholder
			// debugging
			fprintf( stderr, "Error creating program\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( linearKernelKernelProgram, 0, NULL, NULL,
					NULL, NULL );
		if ( 0 != errorCode )
		{
			// TODO:
			// deal with this error
			fprintf( stderr, "Error building program\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( linearKernelKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		linearKernelKernel = clCreateKernel( linearKernelKernelProgram, "linear_kernel_kernel", &errorCode );
		if ( 0 != errorCode )
		{
			fprintf( stderr, "Error creating linear kernel kernel\n" );
			exit( -1 );
		}
		
		customDaxpyKernelProgram = clCreateProgramWithSource( kernelContext,
								1,
								&customDaxpyKernelSource,
								NULL,
								&errorCode );
		if ( 0 != errorCode )
		{
			// TODO:
			// replace this placeholder
			// debugging
			fprintf( stderr, "Error creating custom daxpy program\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( customDaxpyKernelProgram, 0, NULL, NULL,
					NULL, NULL );
		if ( 0 != errorCode )
		{
			// TODO:
			// deal with this error
			fprintf( stderr, "Error building daxpy program program\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( customDaxpyKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		customDaxpyKernel = clCreateKernel( customDaxpyKernelProgram, "custom_daxpy_kernel", &errorCode );
		if ( 0 != errorCode )
		{
			fprintf( stderr, "Error creating custom daxpy kernel kernel\n" );
			exit( -1 );
		}
		
		//swapObjectiveFunctionKernelProgram = clCreateProgramWithSource( kernelContext,
		swapObjectiveFunctionKernelProgram = clCreateProgramWithSource( kernelCpuContext,
											1,
											&swapObjectiveFunctionKernelSource,
											NULL,
											&errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR CREATING OBJECTIVE FUNCTION SWAPPING KERNEL PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( swapObjectiveFunctionKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING OBJECTIVE FUNCTION SWAPPING PROGRAM\n" );
			exit( -1 );
		}
		swapObjectiveFunctionKernel = clCreateKernel( swapObjectiveFunctionKernelProgram, "swap_objective_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR CREATING KERNEL FOR OBJECTIVE FUNCTION SWAPPING PROGRAM\n" );
			exit( -1 );
		}

		findCandidateIValuesKernelProgram = clCreateProgramWithSource( kernelContext,
																		1,
																		&findCandidateIValuesKernelSource,
																		NULL,
																		&errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR CREATING I CANDIDATE SEARCHING KERNEL PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( findCandidateIValuesKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING I CANDIDATE SEARCHING PROGRAM\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( findCandidateIValuesKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		findCandidateIValuesKernel = clCreateKernel( findCandidateIValuesKernelProgram, "find_candidate_i_values_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR CREATING I CANDIDATE SEARCHING KERNEL\n" );
			exit( -1 );
		}

		findCandidateJValuesKernelProgram = clCreateProgramWithSource( kernelContext,
																		1,
																		&findCandidateJValuesKernelSource,
																		NULL,
																		&errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR CREATING J CANDIDATE SEARCHING KERNEL PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( findCandidateJValuesKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING J CANDIDATE SEARCHING PROGRAM\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( findCandidateJValuesKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		findCandidateJValuesKernel = clCreateKernel( findCandidateJValuesKernelProgram, "find_candidate_j_values_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR CREATING J CANDIDATE SEARCHING KERNEL\n" );
			exit( -1 );
		}
		
		dualDaxpyKernelProgram = clCreateProgramWithSource( kernelCpuContext, 1, &dualDaxpyKernelSource, NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error for real
			fprintf( stderr, "ERROR: FAILED TO CREATE DUAL DAXPY KERNEL PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( dualDaxpyKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR: FAILED TO BUILD DUAL DAXPY PROGRAM\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( findCandidateJValuesKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		dualDaxpyKernel = clCreateKernel( dualDaxpyKernelProgram, "dual_daxpy_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR: FAILED TO CREATE DUAL DAXPY KERNEL\n" );
			exit( -1 );
		}

		customMatrixVectorKernelProgram = clCreateProgramWithSource( kernelContext, 1, &customMatrixVectorKernelSource, NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING MATRIX VECTOR MULTIPLY PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( customMatrixVectorKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING MATRIX VECTOR MULTIPLY KERNEL\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( customMatrixVectorKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		customMatrixVectorKernel = clCreateKernel( customMatrixVectorKernelProgram, "custom_matrix_vector_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING MATRIX VECTOR MULT KERNEL\n" );
			exit( -1 );
		}
		
		customMatrixVectorPolynomialKernelProgram = clCreateProgramWithSource( kernelContext, 1, &customMatrixVectorPolynomialKernelSource,
																				NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING POLYNOMIAL KERNEL PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( customMatrixVectorPolynomialKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING POLYNOMIAL MATRIX VECTOR MULTIPLY KERNEL\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( customMatrixVectorPolynomialKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		customMatrixVectorPolynomialKernel = clCreateKernel( customMatrixVectorPolynomialKernelProgram,
															"custom_matrix_vector_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING POLYNOMIAL KERNEL\n" );
			exit( -1 );
		}
		
		customMatrixVectorSigmoidKernelProgram = clCreateProgramWithSource( kernelContext, 1, &customMatrixVectorSigmoidKernelSource,
																				NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING SIGMOID KERNEL PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( customMatrixVectorSigmoidKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING SIGMOID MATRIX VECTOR MULTIPLY KERNEL\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( customMatrixVectorSigmoidKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		customMatrixVectorSigmoidKernel = clCreateKernel( customMatrixVectorSigmoidKernelProgram,
															"custom_matrix_vector_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING SIGMOID KERNEL\n" );
			exit( -1 );
		}
		
		customMatrixVectorRBFKernelProgram = clCreateProgramWithSource( kernelContext, 1, &customMatrixVectorRBFKernelSource,
																				NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING SIGMOID KERNEL PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( customMatrixVectorRBFKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING SIGMOID MATRIX VECTOR MULTIPLY KERNEL\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( customMatrixVectorRBFKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		customMatrixVectorRBFKernel = clCreateKernel( customMatrixVectorRBFKernelProgram,
															"custom_matrix_vector_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING SIGMOID KERNEL\n" );
			exit( -1 );
		}
		
		swapVectorBlockKernelProgram = clCreateProgramWithSource( kernelContext, 1, &swapVectorBlockKernelSource, NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING SWAP VECTOR BLOCK PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( swapVectorBlockKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING SWAP VECTOR BLOCK KERNEL\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( swapVectorBlockKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		swapVectorBlockKernel = clCreateKernel( swapVectorBlockKernelProgram, "swap_vector_block_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING SWAP VECTOR BLOCK KERNEL\n" );
			exit( -1 );
		}
		
		predictionReductionKernelProgram = clCreateProgramWithSource( kernelContext, 1, &predictionReductionKernelSource, NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING PREDICTION REDUCTION PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( predictionReductionKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING PREDICTION REDUCTION KERNEL\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( predictionReductionKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		predictionReductionKernel = clCreateKernel( predictionReductionKernelProgram, "prediction_reduction_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING PREDICTION REDUCTION KERNEL\n" );
			exit( -1 );
		}
		
		/*
		reductionKernelProgram = clCreateProgramWithSource( kernelContext, 1, &reductionKernelSource, NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING REDUCTION PROGRAM\n" );
			exit( -1 );
		}
		errorCode = clBuildProgram( reductionKernelProgram, 0, NULL, NULL, NULL, NULL );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR BUILDING REDUCTION KERNEL\n" );
			char buildLog[1024];
			size_t size;
			errorCode = clGetProgramBuildInfo( findCandidateJValuesKernelProgram,
										firstDevice,
										CL_PROGRAM_BUILD_LOG,
										sizeof(char) * 1024,
										buildLog,
										&size );
			fprintf( stderr, "Size: %lu\n", size );
			fprintf( stderr, "Build log: %s\n", buildLog );
			exit( -1 );
		}
		reductionKernel = clCreateKernel( reductionKernelProgram, "reduction_kernel", &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "ERROR CREATING REDUCTION KERNEL\n" );
			exit( -1 );
		}
		*/

		resultCl = clCreateBuffer( kernelContext, CL_MEM_WRITE_ONLY, sizeof(double), NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			fprintf( stderr, "Error creating linear kernel result buffer\n" );
			exit( -1 );
		}
		
		if ( clAmdBlasSuccess != clAmdBlasSetup() )
		{
			fprintf( stderr, "ERROR FAILED TO SET UP CL AMD BLAS\n" );
			exit( -1 );
		}
		
		numberOfVectors = l;
		
		// TODO: mask kernel assignment
	#endif

	wideKernelInUse = 0;
	
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
		case LINEAR_OPENCL:
			kernel_function = &Kernel::kernel_linear_opencl;
			break;
		case WIDE_LINEAR_OPENCL:
			wide_kernel_function = &Kernel::wide_kernel_linear_opencl;
			kernel_function = &Kernel::kernel_linear;
			wideKernelInUse = 1;
			break;
		case WIDE_POLY_OPENCL:
			wide_kernel_function = &Kernel::wide_kernel_poly_opencl;
			kernel_function = &Kernel::kernel_poly;
			wideKernelInUse = 1;
			break;
		case WIDE_SIGMOID_OPENCL:
			wide_kernel_function = &Kernel::wide_kernel_sigmoid_opencl;
			kernel_function = &Kernel::kernel_sigmoid;
			wideKernelInUse = 1;
			break;
		case WIDE_RBF_OPENCL:
			wide_kernel_function = &Kernel::wide_kernel_rbf_opencl;
			kernel_function = &Kernel::kernel_rbf;
			wideKernelInUse = 1;
			break;
	}
	
	clone(x,x_,l);

	if(kernel_type == RBF || kernel_type == WIDE_RBF_OPENCL)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
		{
			x_square[i] = dot(x[i],x[i]);
		}	
	}
	else
	{
		x_square = 0;
	}
	
	/*#ifdef CL_SVM
		// create the cl_mem buffer (assume all j vectors have same dimensionality)
		x_data_j = clCreateBuffer( kernelContext, CL_MEM_READ_ONLY, sizeof(double) * l * x[0].dim, NULL, &errorCode );
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this
			fprintf( stderr, "ERROR CREATING BUFFER FOR J VALUES\n" );
			exit( -1 );
		}
		// loop through the j vectors and write them
		errorCode = 0;
		for ( k = 0; k < l; k++ )
		{
			errorCode |= clEnqueueWriteBuffer( kernelCommandQueue, x_data_j, CL_FALSE, k * sizeof(double) * l, sizeof(double) * x[0].dim, x[k].values, 0, NULL, NULL );
		}
		if ( CL_SUCCESS != errorCode )
		{
			// TODO: Deal with this
			fprintf( stderr, "ERROR WRITING J VALUES TO GPU MEMORY\n" );
			switch( errorCode )
			{
				case CL_INVALID_COMMAND_QUEUE:
					fprintf( stderr, "CL_INVALID_COMMAND_QUEUE" );
					break;
				case CL_INVALID_CONTEXT:
					fprintf( stderr, "CL_INVALID_CONTEXT" );
					break;
				case CL_INVALID_MEM_OBJECT:
					fprintf( stderr, "CL_INVALID_MEM_OBJECT" );
					break;
				case CL_INVALID_VALUE:
					fprintf( stderr, "CL_INVALID_VALUE" );
					break;
				case CL_INVALID_EVENT_WAIT_LIST:
					fprintf( stderr, "CL_INVALID_EVENT_WAIT_LIST" );
					break;
				case CL_MEM_OBJECT_ALLOCATION_FAILURE:
					fprintf( stderr, "CL_MEM_OBJECT_ALLOCATION_FAILURE" );
					break;
				case CL_OUT_OF_HOST_MEMORY:
					fprintf( stderr, "CL_OUT_OF_HOST_MEMORY" );
					break;
			};
			//return -1;
			exit( -1 );
		}
	#endif*/
	
	
}

Kernel::~Kernel()
{
	// profiling
	fprintf( stdout, "OBJECTIVE FUNCTION UPDATE TIME:\n" );
	fprintf( stdout, "Time taken from other functions: %lu\n", otherFunctionTime );
	fprintf( stdout, "Time taken getting Q:	%lu\n", gettingQTime );
	fprintf( stdout, "Time taken getting G: %lu\n", gettingGTime );
	fprintf( stdout, "Time taken performing DAXPY: %lu\n", daxpyTime );
	
	fprintf( stdout, "CACHE MISSES: %lu\n", gpuCache.cacheMisses );
	fprintf( stdout, "CACHE INVALIDATIONS: %lu\n", gpuCache.cacheInvalidations );
	fprintf( stdout, "Q CACHE MISSES: %lu\n", gpuQCache.cacheMisses );
	fprintf( stdout, "Q CACHE INVALIDATIONS: %lu\n", gpuQCache.cacheInvalidations );

	if ( 0 != swappingCount )
	{
		fprintf( stdout, "SWAPPING TIME: %llu	COUNT: %llu	PER: %lf\n", swappingTime, swappingCount, ((double)swappingTime)/((double)swappingCount) );
		fprintf( stdout, "RETRIEVING Q: %llu \n", qRetrievingTime );
		fprintf( stdout, "OTHER FUNCTIONS Q: %llu \n", swappingOtherFunctionTime );
		fprintf( stdout, "CACHE SWAPPING Q: %llu \n", qCacheSwappingTime );
		fprintf( stdout, "QUEUE THING AT THE END Q: %llu \n", endQueueTime );
		fprintf( stdout, "CALLING KERNEL Q: %llu \n", kernelEnqueuingTime );
		fprintf( stdout, "SETTING ARGS Q: %llu \n", argSettingTime );
	}
	
	fprintf( stdout, "WIDE KERNEL EXECUTION TIME:\n" );
	fprintf( stdout, "TIME WRITING VECTORS: %llu\n", writingVectorsTime );
	fprintf( stdout, "TIME SETTING UP OUTPUT: %llu\n", outputSetupTime );
	fprintf( stdout, "TIME RUNNING BLAS FUNCTION: %llu\n", blasExecutionTime );
	fprintf( stdout, "TIME READING BACK OUTPUT: %llu\n", outputReadingTime );
	
	#ifdef CL_SVM
		// debugging
		fprintf( stdout, "Deconstructing kernel\n" );
		clAmdBlasTeardown();
		clReleaseMemObject( resultCl );
		//clReleaseMemObject( x_data_j );
		clReleaseContext( kernelContext );
		clReleaseCommandQueue( kernelCommandQueue );
	#endif
	delete[] x;
	delete[] x_square;
	
	// debugging
	fprintf( stdout, "Finished deconstructing kernel\n" );
	
}

#ifdef _DENSE_REP
double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;

	int dim = min(px->dim, py->dim);
	for (int i = 0; i < dim; i++)
	{
		sum += (px->values)[i] * (py->values)[i];
	}
	return sum;
}

double Kernel::dot(const svm_node &px, const svm_node &py)
{
	double sum = 0;

	int dim = min(px.dim, py.dim);
	for (int i = 0; i < dim; i++)
	{
		sum += px.values[i] * py.values[i];
	}
	return sum;
}
#else
double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}
#endif

double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
		case WIDE_LINEAR_OPENCL:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
#ifdef _DENSE_REP
			int dim = min(x->dim, y->dim), i;
			for (i = 0; i < dim; i++)
			{
				double d = x->values[i] - y->values[i];
				sum += d*d;
			}
			for (; i < x->dim; i++)
				sum += x->values[i] * x->values[i];
			for (; i < y->dim; i++)
				sum += y->values[i] * y->values[i];
#else
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
#endif
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
#ifdef _DENSE_REP
			return x->values[(int)(y->values[0])];
#else
			return x[(int)(y->value)].value;
#endif
		default:
			return 0;  // Unreachable 
	}
}

#ifdef CL_SVM
		double Kernel::wide_k_function( const svm_node * x, const svm_node * y,
								const svm_parameter & param, double * svmCoefficients  )
		{
			// variables
			svm_node * xCopy;
			
			// function body
			xCopy = (svm_node*) x;
			// call the inner function
			{
				switch( param.kernel_type )
				{
					case LINEAR:
					case WIDE_LINEAR_OPENCL:
						return prediction_setup_linear( xCopy, svmCoefficients );
					case POLY:
					case WIDE_POLY_OPENCL:
						return prediction_setup_poly( xCopy, svmCoefficients );
					case RBF:
					case WIDE_RBF_OPENCL:
						return prediction_setup_rbf( xCopy, svmCoefficients );
					case SIGMOID:
					case WIDE_SIGMOID_OPENCL:
						return prediction_setup_sigmoid( xCopy, svmCoefficients );
					default:
						return 0.0;
				};
			}
			
			// clean up
			return 0.0;
		}
#endif

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const double *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual double calculate_rho();
	virtual void do_shrinking();
private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);	
};

void Solver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	// TODO: duplicate y swap
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	// duplicate work on the GPU
	if ( 0 != ((QMatrix*)Q)->swap_objective_function( i, j ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR SWAPPING OBJECTIVE FUNCTION ON GPU\n" );
		exit( -1 );
	}
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient()
{
	// debugging
	fprintf( stdout, "Reconstructing\n" );

	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int i,j;
	int nr_free = 0;

	// check duplicated GPU work
	/*if ( 0 != Q->check_objective_function( l, G ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR, OBJECTIVE FUNCTIONS DO NOT MATCH BEFORE RECONSTRUCTION\n" );
		exit( -1 );
	}*/
	
	double * G_dup;
	//G_dup = (double*) malloc( sizeof(double) * l );
	
	//if ( 0 != Q->retrieve_objective_function( active_size, l, &(G_dup[active_size]) ) )
	if ( 0 != ((QMatrix*)Q)->retrieve_objective_function( active_size, l, &(G[active_size]) ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR: FAILED TO RETRIEVE G_DUP\n" );
		exit( -1 );
	}
	
	// debugging
	/*for ( j = active_size; j < l; j++ )
	{
		if ( CL_PRECISION_ERROR < std::abs( G[ j ] - G_dup[ j ] ) )
		{
			fprintf( stderr, "AT INDEX %i, GPU AND CPU DATA FOR OBJECTIVE FUNCTION DO NOT MATCH ON CPU SIDE: %lf != %lf\n", j, G[ j ], G_dup[ j ] );
			exit( -1 );
		}
	}*/
	
	for(j=active_size;j<l;j++)
	{
		G[j] = G_bar[j] + p[j];
		//G_dup[j] = G_bar[j] + p[j];
	}
	
	for(j=0;j<active_size;j++)
	{
		if(is_free(j))
		{
			nr_free++;
		}
	}

	if(2*nr_free < active_size)
	{
		info("\nWARNING: using -h 0 may be faster\n");
	}
	
	if (nr_free*l > 2*active_size*(l-active_size))
	{
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i = Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
			{
				if(is_free(j))
				{
					G[i] += alpha[j] * Q_i[j];
					//G_dup[i] += alpha[j] * Q_i[j];
				}
			}
		}
	}
	else
	{
		for(i=0;i<active_size;i++)
		{
			if(is_free(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l);
				double alpha_i = alpha[i];
				for(j=active_size;j<l;j++)
				{
					G[j] += alpha_i * Q_i[j];
					//G_dup[j] += alpha_i * Q_i[j];
				}
			}
		}
	}
	
	// duplicate work on GPU
	//if ( 0 != Q->set_objective_function( active_size, l, &(G_dup[active_size]) ) )
	if ( 0 != ((QMatrix*)Q)->set_objective_function( active_size, l, &(G[active_size]) ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR SETTING G_DUP DATA\n" );
		exit( -1 );
	}
	
	// check duplicated GPU work
	/*if ( 0 != Q->check_objective_function( l, G ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR, OBJECTIVE FUNCTIONS DO NOT MATCH AFTER RECONSTRUCTION\n" );
		exit( -1 );
	}*/
	
	/*if ( 0 != Q->initialize_objective_function( l, G ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR REINITIALIZING OBJECTIVE FUNCTION ON GPU\n" );
		exit( -1 );
	}*/
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
{

	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;
	profileDecls;
	LONGLONG wssTime = 0;
	LONGLONG reconstructionTime = 0;
	LONGLONG kernelMatrixTime = 0;
	LONGLONG objectiveFunctionUpdateTime = 0;
	LONGLONG pureCommunicationTime = 0;
	LONGLONG serialGTime = 0;
	LONGLONG wssCount = 0;
	LONGLONG reconstructionCount = 0;
	LONGLONG kernelMatrixCount = 0;
	LONGLONG objectiveFunctionUpdateCount = 0;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
		{
			update_alpha_status(i);
		}
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
		{
			active_set[i] = i;
		}
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_bar[i] = 0;
		}
		
		// duplicate work on GPU
		startTimer();
		if ( 0 != ((QMatrix*)(&Q))->initialize_objective_function( l, p ) )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR INITIALIZING GPU OBJECTIVE FUNCTION\n" );
			exit( -1 );
		}
		stopTimer();
		pureCommunicationTime += calculateTime();
		if ( 0 != ((QMatrix*)(&Q))->check_objective_function( active_size, G ) )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR, OBJECTIVE FUNCTIONS DO NOT MATCH AFTER ASSIGNING P\n" );
			exit( -1 );
		}
		for(i=0;i<l;i++)
		{
			if(!is_lower_bound(i))
			{
				startTimer();
				const Qfloat *Q_i = Q.get_Q(i,l);
				stopTimer();
				kernelMatrixTime += calculateTime();
				kernelMatrixCount++;
				double alpha_i = alpha[i];
				int j;
				startTimer();
				for(j=0;j<l;j++)
				{
					G[j] += alpha_i*Q_i[j];
				}
				stopTimer();
				serialGTime += calculateTime();
				// duplicate work on GPU
				startTimer();
				if ( 0 != ((QMatrix*)(&Q))->update_objective_function( l, alpha_i, 0.0, i, i ) )
				{
					// TODO: Deal with this error
					fprintf( stderr, "ERROR UPDATING GPU OBJECTIVE FUNCTION\n" );
					exit( -1 );
				}
				stopTimer();
				objectiveFunctionUpdateTime += calculateTime();
				objectiveFunctionUpdateCount++;
				
				if(is_upper_bound(i))
				{
					for(j=0;j<l;j++)
					{
						G_bar[j] += get_C(i) * Q_i[j];
					}
				}
			}
		}
	}

	// check duplicated GPU work
	/*if ( 0 != Q.check_objective_function( active_size, G ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR, OBJECTIVE FUNCTIONS DO NOT MATCH AT INITIALIZATION\n" );
		exit( -1 );
	}
	fprintf( stdout, "PASSED CHECK AT INITIALIZATION\n" );*/
	
	// optimization step
	
	int iter = 0;
	int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	int counter = min(l,1000)+1;

	while(iter < max_iter)
	{
		// show progress and do shrinking
		
		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		startTimer();
		if(select_working_set(i,j)!=0)
		{
			stopTimer();
			wssTime += calculateTime();
			wssCount++;
		
			// reconstruct the whole gradient
			startTimer();
			reconstruct_gradient();
			stopTimer();
			reconstructionTime += calculateTime();
			reconstructionCount++;
			
			// AT THIS POINT WE CAN KEEP A COPY OF G AT LITTLE TO NO COST
			// reset active set size and check
			active_size = l;
			info("*");
			startTimer();
			if(select_working_set(i,j)!=0)
			{
				stopTimer();
				wssTime += calculateTime();
				wssCount++;
				break;
			}
			else
			{
				stopTimer();
				wssTime += calculateTime();
				wssCount++;
				counter = 1;	// do shrinking next iteration
			}
		}
		else
		{
			stopTimer();
			wssTime += calculateTime();
			wssCount++;
		}
		
		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully
		
		startTimer();
		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);
		stopTimer();
		kernelMatrixTime += calculateTime();
		kernelMatrixCount += 2;

		
		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if(y[i]!=y[j])
		{
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			// MARK: CPU USE OF G
			double delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;
			
			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
			{
				quad_coef = TAU;
			}
			// MARK: CPU USE OF G
			double delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		
		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		// this can be optimized iff the Q and G values are stored (possibly mirrored) on the GPU
		startTimer();
		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}
		stopTimer();
		serialGTime += calculateTime();
		// duplicate work on the GPU
		startTimer();
		if ( 0 != ((QMatrix*)(&Q))->update_objective_function( active_size, delta_alpha_i, delta_alpha_j, i, j ) )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR UPDATING OBJECTIVE FUNCTION ON GPU MID ITERATION: %i, %i\n", i, j );
			exit( -1 );
		}
		stopTimer();
		objectiveFunctionUpdateTime += calculateTime();
		objectiveFunctionUpdateCount++;
		
		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				startTimer();
				Q_i = Q.get_Q(i,l);
				stopTimer();
				kernelMatrixTime += calculateTime();
				kernelMatrixCount++;
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				startTimer();
				Q_j = Q.get_Q(j,l);
				stopTimer();
				kernelMatrixTime += calculateTime();
				kernelMatrixCount++;
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}


		// check duplicated GPU work
		/*if ( 0 != Q.check_objective_function( active_size, G ) )
		{
			// TODO: Deal with this error
			fprintf( stderr, "ERROR, OBJECTIVE FUNCTIONS DO NOT MATCH AT END OF ITERATION %i\n", iter );
			exit( -1 );
		}*/
		
	}

	// check duplicated GPU work
	/*if ( 0 != Q.check_objective_function( l, G ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR, OBJECTIVE FUNCTIONS DO NOT MATCH RIGHT AFTER OPTIMIZATION LOOP: %i\n", l );
		exit( -1 );
	}*/
	
	if(iter >= max_iter)
	{
		if(active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
			startTimer();
			reconstruct_gradient();
			stopTimer();
			reconstructionTime += calculateTime();
			reconstructionCount++;
			active_size = l;
			info("*");
		}
		fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
	}

	// check duplicated GPU work
	/*if ( 0 != Q.check_objective_function( l, G ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR, OBJECTIVE FUNCTIONS DO NOT MATCH RIGHT AFTER RECONSTRUCTING GRADIENT\n" );
		exit( -1 );
	}*/
	
	// calculate rho

	// HERE: READ G BACK TO CPU, CONSIDER MOVING RHO CALC AND V CALC TO GPU
	
	startTimer();
	if ( 0 != ((QMatrix*)(&Q))->retrieve_objective_function( 0, l, G ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "FAILED TO RETRIEVE G FUNCTION AT END OF SMO\n" );
		exit( -1 );
	}
	stopTimer();
	pureCommunicationTime += calculateTime();
	
		// REPORT PROFILING
	fprintf( stdout, "PROFILING RESULT:S\n" );
	fprintf( stdout, "WSS	TIME: %llu, COUNT: %llu	PER: %lf\n", wssCount, wssTime, ((double)wssTime)/((double)wssTime) );
	fprintf( stdout, "RECON	TIME: %llu, COUNT: %llu	PER: %lf\n", reconstructionTime, reconstructionCount, ((double)reconstructionTime)/((double)reconstructionCount) );
	fprintf( stdout, "KM	TIME: %llu, COUNT: %llu	PER: %lf\n", kernelMatrixTime, kernelMatrixCount, ((double)kernelMatrixTime)/((double)kernelMatrixCount) );
	fprintf( stdout, "OBJ	TIME: %llu, COUNT: %llu	PER: %lf\n", objectiveFunctionUpdateTime, objectiveFunctionUpdateCount, ((double)objectiveFunctionUpdateTime)/((double)objectiveFunctionUpdateCount) );
	fprintf( stdout, "COMM	TIME: %llu\n", pureCommunicationTime );
	fprintf( stdout, "SG	TIME: %llu\n", serialGTime );
	
	// MARK: CPU USE OF G
	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		// MARK: CPU USE OF G
		for(i=0;i<l;i++)
		{
			v += alpha[i] * (G[i] + p[i]);
		}

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	// In cuSVM, this part is broken up in two levels. 
	// Each block (255 threads) is tasked with selecting
	// the maximum of 255 values, then the result is
	// reduced on the GPU. 
	
	// G is already on the GPU
	// we need to communicate alpha_status
	// also need to communicate y vector, but we could do that at program start? Only if we update its swaps
	
	// this is maximum selection
	
	// find i that maximizes gradient
	{
	
		for(int t=0;t<active_size;t++)
		{
			if(y[t]==+1)	
			{
				if(!is_upper_bound(t))
				{
					if(-G[t] >= Gmax)
					{
						Gmax = -G[t];
						Gmax_idx = t;
					}
				}
			}
			else
			{
				if(!is_lower_bound(t))
				{
					if(G[t] >= Gmax)
					{
						Gmax = G[t];
						Gmax_idx = t;
					}
				}
			}
		}

	}
	
	// save it
	int i = Gmax_idx;
	
	// test duplicate work
	int dup_i;
	double dup_gmax;
	/*if ( 0 != Q->select_working_set_i( y, alpha_status, active_size, dup_i, dup_gmax ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR SELECTING I ON THE GPU\n" );
		exit( -1 );
	}*/
	
	//int i = dup_i;
	//Gmax_idx = i;
	//Gmax = dup_gmax;
	
	/*if ( i != dup_i )// || Gmax != dup_gmax )
	{
		fprintf( stderr, "CPU AND GPU I INDICES DO NOT MATCH: %i != %i, %lf != %lf\n", i, dup_i, Gmax, dup_gmax );
		exit( -1 );
	}
	*/
	const Qfloat *Q_i = NULL;
	
	// get Q_i
	{
		if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		{
			Q_i = Q->get_Q(i,active_size);
		}
	}

	// find j that minimizes decrease in objective value
	{
	
		// for GPU: y, alpha_status, Gmax, QD, i_selection (define TAU on GPU)
		// this is going to be another block based minimization
		for(int j=0;j<active_size;j++)
		{
			// we'll have to write y out to the GPU again, should really cache that
			if(y[j]==+1)
			{
				// have to write alpha_status out to the GPU, should cache that maybe
				if (!is_lower_bound(j))
				{
					// G is already on the GPU, Gmax doesn't take a lot to write
					double grad_diff=Gmax+G[j];
					// Gmax2 can be created on the GPU
					if (G[j] >= Gmax2)
					{
						Gmax2 = G[j];
					}
					if (grad_diff > 0)
					{
						double obj_diff;
						// have to push QD out to the GPU (Q_i should already be there)
						double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
						if (quad_coef > 0)
						{
							obj_diff = -(grad_diff*grad_diff)/quad_coef;
						}
						else
						{
							// should push TAU as a definition? Maybe an argument?
							obj_diff = -(grad_diff*grad_diff)/TAU;
						}
						// obj_diff_min can be calculate entirely on the GPU
						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx=j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
			else
			{
				// have to write alpha_status out to the GPU, should cache that maybe
				if (!is_upper_bound(j))
				{
					// G is already on the GPU, Gmax doesn't take a lot to write
					double grad_diff= Gmax-G[j];
					if (-G[j] >= Gmax2)
					{
						Gmax2 = -G[j];
					}
					if (grad_diff > 0)
					{
						double obj_diff; 
						double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
						if (quad_coef > 0)
						{
							obj_diff = -(grad_diff*grad_diff)/quad_coef;
						}
						else
						{
							obj_diff = -(grad_diff*grad_diff)/TAU;
						}

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx=j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
		}
	}

	int dup_j;
	double gmax2_dup;
	
	// check duplicated GPU work
	/*if ( 0 != Q->check_objective_function( l, G ) )
	{
		// TODO: Deal with this error
		fprintf( stderr, "ERROR, OBJECTIVE FUNCTIONS DO NOT MATCH IN WORKING SET SELECTION AFTER J\n" );
		exit( -1 );
	}*/
	
	
	/*if ( 0 != Q->select_working_set_j( y, alpha_status, Gmax, i, QD, active_size, dup_j, gmax2_dup, G ) )
	{
		fprintf( stderr, "ERROR WHILE SELECTING J ON GPU\n" );
		exit( -1 );
	}
	
	Gmin_idx = dup_j;
	Gmax2 = gmax2_dup;
	*/
	/*if ( dup_j != Gmin_idx && (CL_PRECISION_ERROR < fabs( Gmax2 - gmax2_dup ) ) )
	{
		fprintf( stderr, "ERROR: CPU AND GPU J DO NOT MATCH: %i != %i, %0.20lf != %0.20lf\n", dup_j, Gmin_idx, Gmax2, gmax2_dup );
		exit( -1 );
	}
	*/
	if(Gmax+Gmax2 < eps)
	{
		return 1;
	}

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax1);
	}
	else
		return(false);
}


// MARK: CPU USE OF G
void Solver::do_shrinking()
{
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

// MARK: CPU USE OF G
double Solver::calculate_rho()
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++)
	{
		double yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU: public Solver
{
public:
	Solver_NU() {}
	void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
	{
		this->si = si;
		Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp)
				{
					Gmaxp = -G[t];
					Gmaxp_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn)
				{
					Gmaxn = G[t];
					Gmaxn_idx = t;
				}
		}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in = Q->get_Q(in,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))	
			{
				double grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else	
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU::do_shrinking()
{
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{	
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver_NU::calculate_rho()
{
	int nr_free1 = 0,nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for(int i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(is_upper_bound(i))
				lb1 = max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2 = max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = min(ub2,G[i]);
			else
			{
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;
	
	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;
	
	si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
			{
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
			}
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	double *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
	
		// debugging
		fprintf( stdout, "Constructing one class\n" );
	
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
		{
			QD[i] = (this->*kernel_function)(i,i);
		}
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		// debugging
		//fprintf( stdout, "Getting Q\n" );
		Qfloat *data;
		double * doubleData;
		int start, j;
		// we can insert parallelism here
		// turn off caching for now
		if((start = cache->get_data(i,&data,len)) < len)
		//data = (Qfloat*) malloc( sizeof(Qfloat) * len );
		//start = 0;
		{
			#ifdef CL_SVM
				if ( wideKernelInUse )
				{
					doubleData = (double*) malloc( sizeof(double) * ( numberOfVectors ) );//( len ) );//-start + 1 ) );
					if ( NULL == doubleData )
					{
						// TODO: Deal with this for real
						fprintf( stderr, "FAILED TO ALLOCATE SPACE FOR DOUBLE DATA\n" );
						exit( -1 );
					}
					// after call to get_data, data has been allocated
					if ( -1 == (this->*wide_kernel_function)( i, 0/*start*/, numberOfVectors-1/*len-1*/, doubleData ) )
					{
						// TODO: Deal with this error
						fprintf( stderr, "FAILED TO EXECUTE WIDE KERNEL FUNCTION\n" );
						exit( -1 );
					}
					for ( j = start; j < len; j++ )
					{
						data[ j ] = (Qfloat)( doubleData[ j/* - start*/] );//- start ] );
					}
					free( doubleData );
				}
			else
			#endif
			{
				for(j=start;j<len;j++)
				{
					data[j] = (Qfloat)(this->*kernel_function)(i,j);
				}
			}
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		delete[] QD;
	}
private:
	Cache *cache;
	double *QD;
};

class SVR_Q: public Kernel
{ 
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
		QD = new double[2*l];
		sign = new schar[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k] = (this->*kernel_function)(k,k);
			QD[k+l] = QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int i, int j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int j, real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
		{
			for(j=0;j<l;j++)
				data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
		}

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(j=0;j<len;j++)
			buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
		return buf;
	}

	double *get_QD() const
	{
		return QD;
	}

	~SVR_Q()
	{
		delete cache;
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
	}
private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

#ifdef CL_SVM
	class PREDICTION_Q: public Kernel
	{
		public:
			PREDICTION_Q(int l, svm_node * x_, const svm_parameter& param)
			:Kernel( l, x_, param )
			{
			}
			
			Qfloat * get_Q( int i, int len ) const
			{
				return NULL;
			}
			
			double * get_QD() const
			{
				return NULL;
			}
			
			void swap_index( int i, int j ) const
			{
			}
			
			~PREDICTION_Q()
			{
			}
			
	};
#endif

//
// construct and solve various formulations
//

static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

	Solver s;
	s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int i;
	int l = prob->l;
	double nu = param->nu;

	schar *y = new schar[l];

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	double sum_pos = nu*l/2;
	double sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros = new double[l];

	for(i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	double r = si->r;

	info("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	delete[] y;
	delete[] zeros;
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *zeros = new double[l];
	schar *ones = new schar[l];
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
	{
		alpha[i] = 1;
	}
	if(n<prob->l)
	{
		alpha[n] = param->nu * prob->l - n;
	}
	for(i=n+1;i<l;i++)
	{
		alpha[i] = 0;
	}

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s;
	s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	Solver s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double C = param->C;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	double sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	Solver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

//
// decision_function
//
struct decision_function
{
	double *alpha;
	double rho;	
};

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double,prob->l);
	Solver::SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels, 
	double& A, double& B)
{
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter; 
	
	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*A+B;
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB = dec_values[i]*A+B;
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step)
		{
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;
	
	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;
		
		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Cross-validation decision values for probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB)
{
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int,prob->l);
	double *dec_values = Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++)
	{
		int j = i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++)
	{
		int begin = i*prob->l/nr_fold;
		int end = (i+1)*prob->l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l = prob->l-(end-begin);
#ifdef _DENSE_REP
		subprob.x = Malloc(struct svm_node,subprob.l);
#else
		subprob.x = Malloc(struct svm_node*,subprob.l);
#endif
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = -1;
		else
		{
			svm_parameter subparam = *param;
			subparam.probability=0;
			subparam.C=1.0;
			subparam.nr_weight=2;
			subparam.weight_label = Malloc(int,2);
			subparam.weight = Malloc(double,2);
			subparam.weight_label[0]=+1;
			subparam.weight_label[1]=-1;
			subparam.weight[0]=Cp;
			subparam.weight[1]=Cn;
			struct svm_model *submodel = svm_train(&subprob,&subparam);
			for(j=begin;j<end;j++)
			{
#ifdef _DENSE_REP
				svm_predict_values(submodel,(prob->x+perm[j]),&(dec_values[perm[j]])); 
#else
				svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]])); 
#endif
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}		
			svm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.x);
		free(subprob.y);
	}		
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Return parameter of a Laplace distribution 
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param)
{
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double,prob->l);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	svm_cross_validation(prob,&newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae += fabs(ymv[i]);
	}		
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std) 
			count=count+1;
		else 
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);	
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set. 
	// However, for two-class sets with -1/+1 labels and -1 appears first, 
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}
	
	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->sv_coef = Malloc(double *,1);

		if(param->probability && 
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA = Malloc(double,1);
			model->probA[0] = svm_svr_probability(prob,param);
		}

		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
#ifdef _DENSE_REP
		model->SV = Malloc(svm_node,nSV);
#else
		model->SV = Malloc(svm_node *,nSV);
#endif
		model->sv_coef[0] = Malloc(double,nSV);
		model->sv_indices = Malloc(int,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i+1;
				++j;
			}		

		free(f.alpha);
	}
	else
	{
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
#ifdef _DENSE_REP
		svm_node *x = Malloc(svm_node,l);
#else
		svm_node **x = Malloc(svm_node *,l);
#endif
		int i;
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{	
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
#ifdef _DENSE_REP
				sub_prob.x = Malloc(svm_node,sub_prob.l);
#else
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
#endif
				sub_prob.y = Malloc(double,sub_prob.l);
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class = nr_class;
		
		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];
		
		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		if(param->probability)
		{
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{	
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}
		
		info("Total nSV = %d\n",total_sv);

		model->l = total_sv;
#ifdef _DENSE_REP
		model->SV = Malloc(svm_node,total_sv);
#else
		model->SV = Malloc(svm_node *,total_sv);
#endif
		model->sv_indices = Malloc(int,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i])
			{
				model->SV[p] = x[i];
				model->sv_indices[p++] = perm[i] + 1;
			}
			
		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];
				
				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}
		
		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
	}
	return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++)
			{
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++)
			{
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);	
		free(label);
		free(count);	
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
#ifdef _DENSE_REP
		subprob.x = Malloc(struct svm_node,subprob.l);
#else
		subprob.x = Malloc(struct svm_node*,subprob.l);
#endif
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob,param);
		if(param->probability && 
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
#ifdef _DENSE_REP
				target[perm[j]] = svm_predict_probability(submodel,(prob->x + perm[j]),prob_estimates);
#else
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
#endif
			free(prob_estimates);			
		}
		else
			for(j=begin;j<end;j++)
#ifdef _DENSE_REP
				target[perm[j]] = svm_predict(submodel,prob->x+perm[j]);
#else
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
#endif
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}		
	free(fold_start);
	free(perm);	
}


int svm_get_svm_type(const svm_model *model)
{
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

void svm_get_sv_indices(const svm_model *model, int* indices)
{
	if (model->sv_indices != NULL)
		for(int i=0;i<model->l;i++)
			indices[i] = model->sv_indices[i];
}

int svm_get_nr_sv(const svm_model *model)
{
	return model->l;
}

double svm_get_svr_probability(const svm_model *model)
{
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
	    model->probA!=NULL)
		return model->probA[0];
	else
	{
		fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

#ifdef CL_SVM
	Kernel * predictionKernel = NULL;
	int svm_teardown_prediction()
	{
		// variables
		
		// function body
		if ( NULL != predictionKernel )
		{
			delete predictionKernel;
		}
		
		// clean up
		return 0;
	}
#endif

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	int i;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		
		#ifdef CL_SVM
			if ( NULL == predictionKernel )
			{
				// TODO: Figure out where to free this
				predictionKernel = new PREDICTION_Q(model->l, model->SV, model->param);
			}
			sum = predictionKernel->wide_k_function( x, model->SV, model->param, model->sv_coef[0] );
		#else
			for(i=0;i<model->l;i++)
			{
#ifdef _DENSE_REP
				sum += sv_coef[i] * Kernel::k_function(x,model->SV+i,model->param);
#else
				sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
#endif
			}
		#endif
		sum -= model->rho[0];
		*dec_values = sum;

		if(model->param.svm_type == ONE_CLASS)
		{
			return (sum>0)?1:-1;
		}
		else
		{
			return sum;
		}
	}
	else
	{
		int i;
		int nr_class = model->nr_class;
		int l = model->l;
		
		double *kvalue = Malloc(double,l);
		for(i=0;i<l;i++)
		{
#ifdef _DENSE_REP
			kvalue[i] = Kernel::k_function(x,model->SV+i,model->param);
#else
			kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);
#endif
		}

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];
				
				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		dec_values = Malloc(double, 1);
	}
	else 
	{
		dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	}
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_probability(
	const svm_model *model, const svm_node *x, double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);	     
		return model->label[prob_max_idx];
	}
	else 
		return svm_predict(model, x);
}

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed", "", "linear", "polynomial", "rbf", "sigmoid", NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");
	
	const svm_parameter& param = model->param;

	fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY || param.kernel_type == WIDE_POLY_OPENCL)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID || param.kernel_type == WIDE_POLY_OPENCL || param.kernel_type == WIDE_SIGMOID_OPENCL || param.kernel_type == WIDE_RBF_OPENCL )
		fprintf(fp,"gamma %g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID || param.kernel_type == WIDE_POLY_OPENCL || param.kernel_type == WIDE_SIGMOID_OPENCL )
		fprintf(fp,"coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);
	
	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->rho[i]);
		fprintf(fp, "\n");
	}
	
	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probB[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
#ifdef _DENSE_REP
	const svm_node *SV = model->SV;
#else
	const svm_node * const *SV = model->SV;
#endif

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",sv_coef[j][i]);

#ifdef _DENSE_REP
		const svm_node *p = (SV + i);

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->values[0]));
		else
			for (int j = 0; j < p->dim; j++)
			{
				//if (p->values[j] != 0.0)
				if ( 0 != j )
				{
					fprintf(fp,"%d:%.8g ",j, p->values[j]);
				}
			}
#else
		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
#endif
		fprintf(fp, "\n");
	}
	
	setlocale(LC_ALL, old_locale);
	free(old_locale);
	
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;
	
	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");
	
	// read parameters

	svm_model *model = Malloc(svm_model,1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				
				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				
				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			
			setlocale(LC_ALL, old_locale);
			free(old_locale);
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

#ifdef _DENSE_REP
	int max_index = 1;
	// read the max dimension of all vectors
	while(readline(fp) != NULL)
	{
		char *p;
		p = strrchr(line, ':');
		if(p != NULL)
		{			
			while(*p != ' ' && *p != '\t' && p > line)
				p--;
			if(p > line)
				max_index = (int) strtol(p,&endptr,10) + 1;
		}		
		if(max_index > elements)
		{
			elements = max_index;
		}
	}
#else
	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

#endif
	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);

#ifdef _DENSE_REP
	int index;
	model->SV = Malloc(svm_node,l);

	for(i=0;i<l;i++)
	{
		readline(fp);

		model->SV[i].values = Malloc(double, elements);
		model->SV[i].dim = 0;

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		int *d = &(model->SV[i].dim);
		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			index = (int) strtol(idx,&endptr,10);
			while (*d < index)
			{
				model->SV[i].values[(*d)++] = 0.0;
			}
			model->SV[i].values[(*d)++] = strtod(val,&endptr);
		}
		//model->SV[i].dim--;
	}
#else
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);
	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
#endif
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);
	
	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
	if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
#ifdef _DENSE_REP
	for (int i = 0; i < model_ptr->l; i++)
		free (model_ptr->SV[i].values);
#else
		free((void *)(model_ptr->SV[0]));
#endif
	if(model_ptr->sv_coef)
	{
		for(int i=0;i<model_ptr->nr_class-1;i++)
			free(model_ptr->sv_coef[i]);
	}
	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label= NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB= NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void svm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR)
		return "unknown svm type";
	
	// kernel_type, degree
	
	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED &&
	   kernel_type != WIDE_LINEAR_OPENCL &&
	   kernel_type != WIDE_POLY_OPENCL &&
	   kernel_type != WIDE_SIGMOID_OPENCL &&
	   kernel_type != WIDE_RBF_OPENCL)
		return "unknown kernel type";

	if(param->gamma < 0)
		return "gamma < 0";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";

	if(param->probability == 1 &&
	   svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";


	// check whether nu-svc is feasible
	
	if(svm_type == NU_SVC)
	{
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++)
		{
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label = (int *)realloc(label,max_nr_class*sizeof(int));
					count = (int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}
	
		for(i=0;i<nr_class;i++)
		{
			int n1 = count[i];
			for(int j=i+1;j<nr_class;j++)
			{
				int n2 = count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA!=NULL && model->probB!=NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}

