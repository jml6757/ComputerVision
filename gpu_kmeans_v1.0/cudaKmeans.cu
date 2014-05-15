/*
 * File:
 * cudaKmeans.cu
 * version 1.0
 *
 * DESCRIPTION:
 * This is an implementation of k-means clustering in NVIDIA's CUDA.
 * It is designed to interface with MATLAB to speed up MATLAB's computations.
 * 
 * AUTHOR:
 * Nikolaos Sismanis
 *
 * Aristotle University of Thessaloniki
 * Faculty of Engineering
 * Department of Electrical and Computer Engineering
 * Computer Archtecture Lab
 * 
 * DATE:
 * Jan 2011
 *
 * CONTACT INFO:
 * e-mail: nik_sism@hotmail.com nsismani@auth.gr
 */


#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "cuda.h"

#define max_iterations 50
#define BlockSize 512
#define NUMBER_OF_SUBMATRICES 128


typedef struct{
  float *dataset;
  float *members;
  int leading_dim;
  int secondary_dim;
} data_struct;

const char *help="Error using cudaKmeans: 4 arguments\n"
"First (single): dataset, an [mxn] matrix. m must be a multiplier of 128 and n equal or smaller of 512\n"
"Second (single): dataset' ,the transposed dataset an [nxm] matrix\n"
"Third (double): The number of clusters\n"
"Fourth (single): The starting centers (optional)\n";

void errorMessage1(int numarg){

  if(numarg < 3){
    mexErrMsgTxt(help);
  }

}

void errorMessage2(int Objects, int attributes, int numCluster){

  if((Objects % NUMBER_OF_SUBMATRICES) != 0 || attributes > 512 || numCluster >= Objects){
    mexErrMsgTxt(help);
  }
}

void initialize_clusters_rand(data_struct *data_in, data_struct *cluster_in){

  int i, pick = 0;

  int n = cluster_in->leading_dim;
  int m = cluster_in->secondary_dim;
  int Objects = data_in->secondary_dim;
  float *tmp_Centroids = cluster_in->dataset;
  float *tmp_dataset = data_in->dataset;

  srand(time(NULL));
  /*randomly pick initial cluster centers*/
  for(i=0; i<m; i++){
    pick = rand() % Objects;
    tmp_Centroids = cluster_in->dataset + i*n;
    tmp_dataset = data_in->dataset + pick*n;
    memcpy(tmp_Centroids, tmp_dataset, n*sizeof(float));
  }

}

void initialize_clusters(float *buff, data_struct *clusters){

  memcpy(clusters->dataset, buff, clusters->leading_dim*clusters->secondary_dim*sizeof(float));

}

void initialize_device_memory(data_struct *host, data_struct *device){

  cudaMemcpy(device->dataset, host->dataset, host->leading_dim*host->secondary_dim*sizeof(float), cudaMemcpyHostToDevice);

}


void cleanDevice(data_struct *data){

  cudaFree(data->dataset);
  cudaFree(data->members);

}

/* for debug only */
void print(data_struct* data2print){

  int i, j = 0;
  int n = data2print->leading_dim;
  int m = data2print->secondary_dim;
  float *tmp_dataset = data2print->dataset;

  
  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      printf("%f ", tmp_dataset[i*n + j]);
    }
    printf("\n");
  }

  printf("\n");
  
}


__device__ float euclidean_distance_gpu(float *v1, float *v2, int attributes, int numObjects){

  float dist = 0;
  
#pragma unroll 2
  for( int i = 0; i < attributes; i++ )
    {
      float tmp = v2[i*numObjects] - v1[i];
      dist += tmp * tmp;
    }
  return dist;
}


__global__ void CreateClusters(float *dataset, float *centroids, float *index, int *Cond, int numObjects, int numAttributes, int numClusters){

  extern __shared__ float means[];

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int element = tid;
  float tmp, minDist = FLT_MAX;

  int tmp_index = -1;


  if(tid<numObjects){
    for(int center=0; center<numClusters; center++){
      tmp = 0;
      
      if(threadIdx.x<numAttributes){
	means[threadIdx.x] = centroids[center*numAttributes + threadIdx.x];
      }
      __syncthreads();
      
      tmp = euclidean_distance_gpu( means, dataset+element ,numAttributes, numObjects);

      //__syncthreads();
      
      if(tmp < minDist){
	minDist = tmp;
	tmp_index = center;
      }
    }
    //__syncthreads();

    if(index[tid] != tmp_index){
      Cond[0] = 1;
    }

    index[tid] = tmp_index;

  }

}


__global__ void submaticesSum_kernel(float *datasetT, float *Index,
float *centroids, float *ClusterSizes, int numObjects, int numAttributes,
int numClusters){

  
  int tmp_index = 0;
  int submatrix_dim = numObjects/gridDim.x;
  float* block_data = datasetT + blockIdx.x*submatrix_dim*numAttributes;
  float* block_clusterIndex = Index + blockIdx.x*submatrix_dim;
  float* tempBlockSum = centroids + blockIdx.x*numClusters*numAttributes;
  float* tempBlockclusterCount = ClusterSizes + blockIdx.x*numClusters; 

  /*Zero Sum and clusterCount*/
#pragma unroll 2
  for(int i=0; i<numClusters; i++){
    tempBlockSum[i*numAttributes + threadIdx.x] = 0;
    if(threadIdx.x==0){
      tempBlockclusterCount[i] = 0;
    }
    __syncthreads();
  }
  
  //__syncthreads();
#pragma unroll 8 
  for(int i=0; i < submatrix_dim; i++){

    tmp_index = block_clusterIndex[i];
    //__syncthreads();    


    tempBlockSum[tmp_index*numAttributes + threadIdx.x] += block_data[i*numAttributes + threadIdx.x];
    
    if(threadIdx.x==0){
      tempBlockclusterCount[tmp_index]++;
    }
    //__syncthreads();
  }
  
  //__syncthreads();
}


__global__ void calc_ClusterSizes(float *BlockClusterCount, int numCenters){

  float mean_sum = 0;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid<numCenters){

    mean_sum = BlockClusterCount[tid];
    __syncthreads();
#pragma unroll 8
    for(int i=1; i<NUMBER_OF_SUBMATRICES; i++){
      mean_sum += BlockClusterCount[i*numCenters + tid]; 
    }
    //__syncthreads();
    BlockClusterCount[tid] = mean_sum;
  }
}


__global__ void newCentroids(float *centroids, float *clusterSizes, int attributes, int numCenters){ 

  extern __shared__ float attrib[]; 
  
  attrib[threadIdx.x]  = centroids[blockIdx.x*attributes + threadIdx.x];  
  __syncthreads();
  
#pragma unroll 8
  for(int j=1; j<NUMBER_OF_SUBMATRICES; j++){
    attrib[threadIdx.x] += centroids[j*numCenters*attributes + blockIdx.x*attributes + threadIdx.x];
  }

  __syncthreads();

  centroids[blockIdx.x*attributes + threadIdx.x] = attrib[threadIdx.x] / clusterSizes[blockIdx.x]; 


}


void cluster(data_struct *data ,data_struct *clusters, data_struct *dataT){


  int iter = 0;
  int Cond, *d_Cond;
  int numObjects = data->leading_dim;
  int numAttributes = data->secondary_dim;
  int numClusters = clusters->secondary_dim;
  float *dataset = data->dataset;
  float *centroids = clusters->dataset;
  float *Index = data->members;
  float *clusterSizes = clusters->members;
  float *datasetT = dataT->dataset;  


#ifdef TIMEONLY
  float elapsedTime_kernel;
  cudaEvent_t start_kernel, stop_kernel;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
#endif


  cudaMalloc((void**)&d_Cond, sizeof(int));
  
  float tmp_grid_size = ceil((float)numObjects/(float)BlockSize);
  float tmp_block_size = numObjects<BlockSize ? numObjects:BlockSize; 

  dim3 grid((int)tmp_grid_size,1);
  dim3 threads((int)tmp_block_size, 1);

  dim3 submatriceGrid(NUMBER_OF_SUBMATRICES, 1);  
  dim3 submatriceThreads(numAttributes, 1);

  tmp_grid_size = ceil((float)numClusters/(float)BlockSize);
  tmp_block_size = numClusters<BlockSize ? numClusters:BlockSize; 

  dim3 sizeGrid((int)tmp_grid_size, 1);
  dim3 sizeThreads((int)tmp_block_size, 1);

  dim3 newCentroidsGrid(numClusters, 1);
  dim3 newCentroidsThreads(numAttributes, 1);


#ifdef TIMEONLY
  cudaEventRecord(start_kernel, 0);
#endif


  for(iter=0; iter<max_iterations; iter++){

    Cond = 0;
    cudaMemcpy(d_Cond, &Cond, sizeof(int), cudaMemcpyHostToDevice);

    CreateClusters<<<grid, threads, numAttributes*sizeof(float)>>>(dataset, centroids, Index, d_Cond, numObjects, numAttributes, numClusters);

    submaticesSum_kernel<<<submatriceGrid, submatriceThreads>>>(datasetT, Index, centroids, clusterSizes, numObjects, numAttributes, numClusters);

    calc_ClusterSizes<<<sizeGrid, sizeThreads>>>(clusterSizes, numClusters);

    newCentroids<<<newCentroidsGrid, newCentroidsThreads, numAttributes*sizeof(float)>>>(centroids, clusterSizes, numAttributes, numClusters);

    cudaMemcpy(&Cond, d_Cond, sizeof(int), cudaMemcpyDeviceToHost);

    if(Cond == 0){
      /*printf("\nCondition Reached, Process Terminating\n");*/
#ifndef TIMEONLY
      break;
#endif
    }

  }


#ifdef TIMEONLY
  cudaEventRecord(stop_kernel, 0);  
  cudaEventSynchronize(stop_kernel);
#endif


#ifdef TIMEONLY
  printf("\nFinised after %d iterations\n", iter);
#endif

#ifdef TIMEONLY
  cudaEventElapsedTime(&elapsedTime_kernel, start_kernel, stop_kernel);
  printf("Time elapsed for kernel execution: %f ms\n", elapsedTime_kernel);
#endif

  cudaFree(d_Cond);
#ifdef TIMEONLY
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel);
#endif
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){


  /*===== Host ======*/
  data_struct data_in;
  data_struct clusters;
  data_struct data_inT;
  //float SSE=0, new_SSE=0;
  double *numClusters_buf = 0;
  int numClusters = 0;
  float *execTime;

  //ptr2KernelFunction ptr2Func;

  /*==== Device ======*/
  data_struct d_data;
  data_struct d_clusters;
  data_struct d_dataT;

  /*===== Cuda Events===*/
  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  /*  
#ifdef TIMEONLY
  float elapsedTime_kernel;
  cudaEvent_t start_kernel, stop_kernel;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
#endif
  */
  /*======== Initialization=======*/
  errorMessage1(nrhs);

  numClusters_buf = mxGetPr(prhs[2]);
  numClusters = (int)numClusters_buf[0];

  data_in.dataset = (float*)mxGetData(prhs[0]);
  data_in.leading_dim = mxGetM(prhs[0]);
  data_in.secondary_dim = mxGetN(prhs[0]);
  plhs[1] =  mxCreateNumericMatrix(data_in.leading_dim, 1, mxSINGLE_CLASS,mxREAL);
  data_in.members = (float*)mxGetData(plhs[1]);

  data_inT.dataset = (float*)mxGetData(prhs[1]);
  data_inT.leading_dim = mxGetM(prhs[1]);
  data_inT.secondary_dim = mxGetN(prhs[1]); 

  plhs[0] =  mxCreateNumericMatrix(data_in.secondary_dim, numClusters, mxSINGLE_CLASS, mxREAL);
  clusters.dataset = (float*)mxGetData(plhs[0]);
  clusters.leading_dim = data_in.secondary_dim;
  clusters.secondary_dim = numClusters;
  plhs[2] = mxCreateNumericMatrix(numClusters, 1, mxSINGLE_CLASS, mxREAL);
  clusters.members = (float*)mxGetData(plhs[2]);

  d_data.leading_dim = data_in.leading_dim;
  d_data.secondary_dim = data_in.secondary_dim;
  d_clusters.leading_dim = clusters.leading_dim;
  d_clusters.secondary_dim = clusters.secondary_dim;
  d_dataT.leading_dim = data_inT.leading_dim;
  d_dataT.secondary_dim = data_inT.secondary_dim;

  plhs[3] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
  execTime = (float*)mxGetData(plhs[3]);

  errorMessage2(data_in.leading_dim, data_in.secondary_dim, numClusters);

  /*========= device memory allocation======*/

  cudaMalloc((void**)&d_data.dataset, d_data.leading_dim*d_data.secondary_dim*sizeof(float));
  cudaMalloc((void**)&d_data.members, d_data.leading_dim*sizeof(float));

  cudaMalloc((void**)&d_clusters.dataset, d_clusters.leading_dim*d_clusters.secondary_dim*NUMBER_OF_SUBMATRICES*sizeof(float));
  cudaMalloc((void**)&d_clusters.members, d_clusters.secondary_dim*NUMBER_OF_SUBMATRICES*sizeof(float));

  cudaMalloc((void**)&d_dataT.dataset, d_dataT.leading_dim*d_dataT.secondary_dim*sizeof(float));
  cudaMalloc((void**)&d_dataT.members, sizeof(float));


  /* initialize centroids*/
  if(nrhs == 4){
    float *centr_buff = (float*)mxGetData(prhs[3]);
    initialize_clusters(centr_buff, &clusters);
  }
  else{
    initialize_clusters_rand(&data_inT, &clusters);
  }

  cudaEventRecord(start, 0);

  initialize_device_memory(&data_in, &d_data);
  initialize_device_memory(&clusters, &d_clusters);
  initialize_device_memory(&data_inT, &d_dataT);
  /*
#ifdef TIMEONLY
  cudaEventRecord(start_kernel, 0);
#endif
  */
  cluster(&d_data, &d_clusters, &d_dataT);
  /*
#ifdef TIMEONLY
  cudaEventRecord(stop_kernel, 0);  
  cudaEventSynchronize(stop_kernel);
#endif
  */
  cudaMemcpy(clusters.dataset, d_clusters.dataset, clusters.leading_dim*clusters.secondary_dim*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(data_in.members, d_data.members, data_in.leading_dim*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(clusters.members, d_clusters.members, clusters.secondary_dim*sizeof(float), cudaMemcpyDeviceToHost);
  

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);


  cudaEventElapsedTime(&elapsedTime, start, stop);

#ifdef TIMEONLY
  printf("Time elapsed: %f ms\n", elapsedTime);
#endif

  execTime[0] = elapsedTime; 
  /*
#ifdef TIMEONLY
  cudaEventElapsedTime(&elapsedTime_kernel, start_kernel, stop_kernel);
  printf("Time elapsed for kernel execution: %f ms\n", elapsedTime_kernel);
#endif
  */
  /*==== clean device===*/
  cleanDevice(&d_data);
  cleanDevice(&d_clusters);
  cleanDevice(&d_dataT);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  /*
#ifdef TIMEONLY
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel);
#endif
  */
}

