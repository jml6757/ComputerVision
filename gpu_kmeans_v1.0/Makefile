#####################################################
# Makefile for GPU kmeans matlab Toolbox
# version 1.0
#
#
# DATE: Jan 2011
# AUTHOR: Nikolaos Sismanis
# CONTACT INFO: nik_sism@hotmail.com nsismani@auth.gr
#####################################################

# General macros
NVCC=nvcc
MEX=mex
CC=gcc
MATLAB_ROOT=/usr/local/MATLAB/R2010b/
CUDA_INC=/usr/local/cuda/include/
CUDA_LIB=/usr/local/cuda/lib64/

FLG=-O4
NVCC_OP=-Xcompiler -fPIC
CC_OP=-fPIC
MEX_INC=$(MATLAB_ROOT)/extern/include
LIBS=-lcuda -lcudart -lcufft -lcublas -largeArrayDims
SRC=./src
BIN=./bin
OBJ=./obj
DEMO=./demos

mkdircmd=mkdir -p

all: cudaKmeans kmeans_serial cudaKmeans_Timedemo kmeans_serial_Timedemo


cudaKmeans: $(OBJ)/cudaKmeans.o

	$(MEX) -O $(OBJ)/cudaKmeans.o -L$(CUDA_LIB) $(LIBS) -o $(BIN)/cudaKmeans

$(OBJ)/cudaKmeans.o: $(SRC)/cudaKmeans.cu

	$(mkdircmd) $(OBJ)
	 $(NVCC) -c $(FLG) $(SRC)/cudaKmeans.cu $(NVCC_OP) -I$(MEX_INC) -o $(OBJ)/cudaKmeans.o 

kmeans_serial: $(OBJ)/kmeans_serial.o

	$(MEX) -O  $(OBJ)/kmeans_serial.o -L$(CUDA_LIB) $(LIBS)  -o $(BIN)/kmeans_serial

$(OBJ)/kmeans_serial.o: $(SRC)/kmeans_serial.c

	$(CC) -c $(FLG) $(SRC)/kmeans_serial.c -I$(MEX_INC) -I$(CUDA_INC) $(CC_OP) -o $(OBJ)/kmeans_serial.o


cudaKmeans_Timedemo: $(OBJ)/cudaKmeans_Timedemo.o

	$(MEX) -O $(OBJ)/cudaKmeans_Timedemo.o -L$(CUDA_LIB) $(LIBS) -o $(DEMO)/cudaKmeans_Timedemo


$(OBJ)/cudaKmeans_Timedemo.o: $(SRC)/cudaKmeans.cu

	 $(NVCC) -c -D TIMEONLY $(FLG) $(SRC)/cudaKmeans.cu $(NVCC_OP) -I$(MEX_INC) -o $(OBJ)/cudaKmeans_Timedemo.o


kmeans_serial_Timedemo: $(OBJ)/kmeans_serial_Timedemo.o

	$(MEX) -O $(OBJ)/kmeans_serial_Timedemo.o -L$(CUDA_LIB) $(LIBS) -o $(DEMO)/kmeans_serial_Timedemo


$(OBJ)/kmeans_serial_Timedemo.o: $(SRC)/kmeans_serial.c

	$(CC) -c -D TIMEONLY $(FLG) $(SRC)/kmeans_serial.c -I$(MEX_INC) -I$(CUDA_INC) $(CC_OP) -o $(OBJ)/kmeans_serial_Timedemo.o


clean: 
	rm -f $(OBJ)/*o $(BIN)/*mexa* $(DEMO)/*mexa* 
	rm -f ./*~ $(SRC)/*~ $(BIN)/*~ $(DEMO)/*~
	rmdir $(OBJ)

