% Parameters
k = 128;

% Get data from files
display('Loading Data...');

% Get the project directory
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/BuildClusters','');

X = load(strcat(ROOT_DIR,'data/', 'feature_array.dat'),'-mat');
X = X.X;

%Create bag of words using kmeans clustering
display ('Computing k-means...');

opts = statset('Display','iter','UseParallel',true,'MaxIter',1);
[IDX,C] = kmeans(X,k,'Options',opts,'Replicates',8);

% Store extracted feature vector to disk
display ('Saving Data...');

save(strcat(ROOT_DIR,'data/','cluster_index.dat'), 'IDX');
save(strcat(ROOT_DIR,'data/','clusters.dat'), 'C');

display ('Done.');
