% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix

try
	Type = ver;
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		mex libsvmread.c
		mex libsvmwrite.c
		mex svmtrain.c ../svm.cpp svm_model_matlab.c
		mex svmpredict.c ../svm.cpp svm_model_matlab.c
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
	else
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.c
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims -I../include -I../src/opencl_kernels -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include" -I"C:/Program Files (x86)/AMD/clAmdBlas/include" svmtrain.cpp ../src/svm/svm.cpp svm_model_matlab.cpp -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/lib/x64" -lOpenCL -L"C:/Program Files (x86)/AMD/clAmdBlas/lib64/import" -lclAmdBlas 
		% TODO: This part
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims -I../include -I../src/opencl_kernels -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include" -I"C:/Program Files (x86)/AMD/clAmdBlas/include" svmpredict.cpp ../src/svm/svm.cpp svm_model_matlab.cpp  -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/lib/x64" -lOpenCL -L"C:/Program Files (x86)/AMD/clAmdBlas/lib64/import" -lclAmdBlas 
		
	end
catch
	fprintf('If make.m fails, please check README about detailed instructions.\n');
end
