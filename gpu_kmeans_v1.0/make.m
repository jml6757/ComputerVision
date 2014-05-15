try
    nvcc -c -O4 cudaKmeans.cu -Xcompiler -fPIC -I"C:\Program Files\MATLAB\R2013a\extern\include" -o cudaKmeans.o 
    mex -O cudaKmeans.o -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\x64" -lcuda -lcudart -lcufft -lcublas -largeArrayDims

catch
	fprintf('If make.m fails, please check README about detailed instructions.\n');
end
