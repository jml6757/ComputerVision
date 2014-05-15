/*
 * FILE:
 * cudaKmeans.cu
 * version 1.0
 *
 * DESCRIPTION:
 * This is a serial implementation of k-means clustering in C.
 *
 * AUTHOR:
 * Nikolaos Sismanis
 *
 * Aristotle University of Thessaloniki
 * Faculty of Engineering
 * Department of Electrical and Computer Engineering
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
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define max_iterations 50

typedef struct {
  float *dataset;
  float *members;
  unsigned int leading_dim;
  unsigned int secondary_dim; 
} data_struct;


void error_message(){

char *help = "Error using kmeans: Three arguments required\n"
  "First: number of elements\n"
  "Second: number of attributes (dimensions)\n"
  "Third: numder of Clusters\n";

  printf(help);

}

void initialize_clusters(float *buff, data_struct *clusters){

  memcpy(clusters->dataset, buff, clusters->leading_dim*clusters->secondary_dim*sizeof(float));

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
  
}

void clean(data_struct* data1){

  free(data1->dataset);
  free(data1->members);
}


float eucledean_distance(float *v1, float *v2, int length){

  int i = 0;
  float dist = 0;

  for(i=0; i<length; i++){
    dist += (v1[i] - v2[i])*(v1[i] - v2[i]); 
  }

  return(dist);
}


void kmeans_process(data_struct *data_in, data_struct *clusters, float *newCentroids, int* Count){

  int i, j, k;
  float tmp_dist = 0;
  int tmp_index = 0;
  float min_dist = 0;
  float *dataset = data_in->dataset;
  float *centroids = clusters->dataset;
  float *Index = data_in->members;
  float *cluster_size = clusters->members;

  Count[0] = 0;

  for(i=0; i<clusters->secondary_dim; i++){
    cluster_size[i] = 0;
  }

  for(i=0; i<data_in->secondary_dim; i++){
    tmp_dist = 0;
    tmp_index = 0;
    min_dist = FLT_MAX;
    /*find nearest center*/
    for(k=0; k<clusters->secondary_dim; k++){
      tmp_dist = eucledean_distance(dataset+i*data_in->leading_dim, centroids+k*clusters->leading_dim, data_in->leading_dim);
      if(tmp_dist<min_dist){
	min_dist = tmp_dist;
	tmp_index = k;
      }
    }

    if(Index[i] == (float)tmp_index){
      Count[0]++;
    }

    Index[i] = (float)tmp_index;
    cluster_size[tmp_index]++;
    for(j=0; j<data_in->leading_dim; j++){
      newCentroids[tmp_index * clusters->leading_dim + j] += dataset[i * data_in->leading_dim + j]; 
    }
   
  }

  /*update cluster centers*/
  for(k=0; k<clusters->secondary_dim; k++){
    for(j=0; j<data_in->leading_dim; j++){
      centroids[k * clusters->leading_dim + j] = newCentroids[k * clusters->leading_dim + j] / (float)cluster_size[k];

    }
  }

}

void cluster(data_struct *data_in, data_struct *clusters){ 

  int iter, i, j;
  int Count = 0;
  float* newCentroids;


  newCentroids = (float*)malloc(clusters->leading_dim*clusters->secondary_dim*sizeof(float));

  for(iter=0; iter<max_iterations; iter++){

    for(i=0; i<clusters->secondary_dim; i++){
      for(j=0; j<clusters->leading_dim; j++){
	newCentroids[i * clusters->leading_dim + j] = 0;
      }
    }

    kmeans_process(data_in, clusters, newCentroids, &Count);


    if(Count == data_in->secondary_dim){
#ifndef TIMEONLY
      break;
#endif
    }

  }

  printf("Finished after %d iterations\n", iter);

  free(newCentroids);

}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);  

  if(nrhs<2){
    error_message();
  }

  float *execTime;
  int numObjects = mxGetN(prhs[0]);
  int numAttributes = mxGetM(prhs[0]);
  double *numClusters_tmp = (double*)mxGetPr(prhs[1]);
  int numClusters = (int)numClusters_tmp[0];
  int i =0 ;


  data_struct data_in;
  data_struct clusters;
  float *init_centroids;

  /*=======Memory Allocation=========*/

  data_in.leading_dim = mxGetM(prhs[0]);
  data_in.secondary_dim = mxGetN(prhs[0]);
  data_in.dataset = (float*)mxGetData(prhs[0]);
  plhs[1] = mxCreateNumericMatrix(data_in.secondary_dim, 1, mxSINGLE_CLASS, mxREAL);
  data_in.members = (float*)mxGetData(plhs[1]);


  clusters.leading_dim = numAttributes;
  clusters.secondary_dim = numClusters;
  plhs[0] = mxCreateNumericMatrix(clusters.secondary_dim, clusters.leading_dim, mxSINGLE_CLASS, mxREAL);
  clusters.dataset = (float*)mxGetData(plhs[0]);
  plhs[2] = mxCreateNumericMatrix(clusters.secondary_dim, 1, mxSINGLE_CLASS, mxREAL);
  clusters.members = (float*)mxGetData(plhs[2]);

  plhs[3] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
  execTime = (float*)mxGetData(plhs[3]);

  /*=============initialize ==========*/
  if(nrhs == 3){
    init_centroids = (float*)mxGetData(prhs[2]);
    initialize_clusters(init_centroids, &clusters);
  }
  else {
    initialize_clusters_rand(&data_in, &clusters);
  }
  /*=================================*/

  cudaEventRecord(start, 0);

  cluster(&data_in, &clusters);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Time elapsed: %f ms\n", elapsedTime);
  execTime[0] = elapsedTime; 

}
