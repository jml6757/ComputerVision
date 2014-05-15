% Parameters
k = 256;

% Get data from files
display('Loading Data...');

X = load(strcat(DATA_PATH, 'feature_array.dat'),'-mat');
X = X.X;

%Create bag of words using kmeans clustering
display ('Computing k-means...');

[IDX,C] = kmeans(X,k);

% Store extracted feature vector to disk
display ('Saving Data...');

save(strcat(ROOT_DIR,'data/','cluster_index.dat'), 'IDX');
save(strcat(ROOT_DIR,'data/','clusters.dat'), 'C');

display ('Done.');
