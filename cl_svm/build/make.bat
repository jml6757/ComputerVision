python stringify.py < ../code/src/opencl_kernels/linearKernelKernelSource.c > ../code/src/opencl_kernels/linearKernelKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/customDaxpyKernelSource.c > ../code/src/opencl_kernels/customDaxpyKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/swapObjectiveFunctionKernelSource.c > ../code/src/opencl_kernels/swapObjectiveFunctionKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/findCandidateIValuesKernelSource.c > ../code/src/opencl_kernels/findCandidateIValuesKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/findCandidateJValuesKernelSource.c > ../code/src/opencl_kernels/findCandidateJValuesKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/dualDaxpyKernelSource.c > ../code/src/opencl_kernels/dualDaxpyKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/customMatrixVectorKernelSource.c > ../code/src/opencl_kernels/customMatrixVectorKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/customMatrixVectorPolynomialKernelSource.c > ../code/src/opencl_kernels/customMatrixVectorPolynomialKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/customMatrixVectorSigmoidKernelSource.c > ../code/src/opencl_kernels/customMatrixVectorSigmoidKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/customMatrixVectorRBFKernelSource.c > ../code/src/opencl_kernels/customMatrixVectorRBFKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/reductionKernelSource.c > ../code/src/opencl_kernels/reductionKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/swapVectorBlockKernelSource.c > ../code/src/opencl_kernels/swapVectorBlockKernelSource.cl
python stringify.py < ../code/src/opencl_kernels/predictionReductionKernelSource.c > ../code/src/opencl_kernels/predictionReductionKernelSource.cl

mingw32-make

copy svm-train.exe ..\
copy svm-train.exe ..\testing\src\SMOTesting\svm-train_opencl.exe
copy svm-predict.exe ..\testing\src\PredictionTesting\svm-predict_opencl.exe
