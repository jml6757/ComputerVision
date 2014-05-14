
project(CL_SVM)
cmake_minimum_required(VERSION 2.8)


SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fpermissive -w" )

include_directories( code/include )
include_directories( code/src/svm-predict )
include_directories( code/src/svm-train )
include_directories( code/src/svm )
include_directories( code/src/opencl_kernels )
include_directories( "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include" )
#include_directories( "C:/Program Files (x86)/AMD APP SDK/2.9/include" )
include_directories( "C:/Program Files (x86)/AMD/clAmdBlas/include" )
link_directories( "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/lib/Win32" )
#link_directories( "C:/Program Files (x86)/AMD APP SDK/2.9/lib/x86" )
link_directories( "C:/Program Files (x86)/AMD/clAmdBlas/lib32/import" )

#add_library( cross_validation code/src/svm-train/cross_validation_with_matrix_precomputation.c )
#add_library( kernel_matrix code/src/svm-train/kernel_matrix_calculation.c )
add_library( svm_lib code/src/svm/svm.cpp )
add_executable( svm-train code/src/svm-train/svm-train.c )
add_executable( KernelTesting testing/src/KernelTesting/KernelTesting.cpp )
add_executable( blasTesting testing/src/KernelTesting/blasTest.cpp )
add_executable( svm-predict code/src/svm-predict/svm-predict.c )
add_executable( cpuTesting testing/src/OCLTesting/TestCpu.c )

target_link_libraries( svm-train svm_lib )
target_link_libraries( svm-train clAmdBlas )
target_link_libraries( svm-train OpenCL )
target_link_libraries( svm-predict svm_lib )
target_link_libraries( svm-predict clAmdBlas )
target_link_libraries( svm-predict OpenCL )
target_link_libraries( KernelTesting svm_lib )
target_link_libraries( KernelTesting clAmdBlas )
target_link_libraries( KernelTesting OpenCL )

target_link_libraries( cpuTesting OpenCL )

target_link_libraries( blasTesting clAmdBlas )
target_link_libraries( blasTesting OpenCL )

add_custom_command( TARGET svm-train PRE_BUILD COMMAND python stringify.py < ../code/src/opencl_kernels/linearKernelKernelSource.c > ../code/src/opencl_kernels/linearKernelKernelSource.cl )
add_custom_command( TARGET KernelTesting PRE_BUILD COMMAND python stringify.py < ../code/src/opencl_kernels/linearKernelKernelSource.c > ../code/src/opencl_kernels/linearKernelKernelSource.cl )

#IF (WIN32)
#	add_custom_command( TARGET svm-train POST_BUILD COMMAND copy svm-train.exe ..\ )
#ENDIF (WIN32)
#IF (APPLE)
#	add_custom_command( TARGET svm-train POST_BUILD COMMAND cp svm-train ../ )
#ENDIF (APPLE)

#target_link_libraries( svm-train cross_validation )
#target_link_libraries( svm-train kernel_matrix )
