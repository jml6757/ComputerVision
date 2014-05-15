% Parameters
k = 512;

% Get the project directory
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/BuildClusters','');
LIB_PATH  = strcat(ROOT_DIR, 'lib/');
addpath(LIB_PATH);

% Get data from files
display('Loading Data...');
X = load(strcat(ROOT_DIR,'data/', 'feature_array.dat'),'-mat');
X = X.X;

%Create bag of words using kmeans clustering
display ('Computing k-means...');

[C, IDX] = kmeans_gpu(X,k);

% Store extracted feature vector to disk
display ('Saving Data...');

save(strcat(ROOT_DIR,'data/','cluster_index.dat'), 'IDX');
save(strcat(ROOT_DIR,'data/','clusters.dat'), 'C');

display ('Done.');
