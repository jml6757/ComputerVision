tic;
[ labels, data ] = libsvmread( 'C:\StereoImaging\cl_svm\testing\data\SVM_IM_DATA_P_1250_N_10000.txt' );
model = svmtrain( labels, data, '-s 2 -t 6 -h 0' )
n = 100
[ predictedLabels, accuracy, X ] = svmpredict( ones(n,1), data( 1:n, : ), model );

timeTaken = toc;
