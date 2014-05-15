%TrainSVM - Opens the extracted feature sets from data files
%           and trains a support vector machine for the chosen
%           data set.


% Parameters
CATEGORY = 1; % Train with this category as the positive images

% Get the project directory and set up paths
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/TrainSVM','');
DATA_PATH = strcat(ROOT_DIR, 'data/');
LIB_PATH  = strcat(ROOT_DIR, 'lib/');
addpath(DATA_PATH);
addpath(LIB_PATH);

% Load positive data
display ('Loading Data...');
DATA = load(strcat(DATA_PATH, 'image_data.dat'),'-mat');
DATA = DATA.DATA;

% Shuffle data
TRAINING_LABELS = zeros(length(DATA), 1);
TRAINING_FEATURES = zeros(length(DATA), length(DATA(1).histogram));

%Add data to structured SVM training arrays
display ('Formatting Data...');
for i = 1:length(DATA)
    if(DATA(i).category == CATEGORY)
        LABEL = 1;
    else
        LABEL = -1;
    end
    TRAINING_LABELS(i) =  LABEL;
    TRAINING_FEATURES(i) = POS_DATA(i).histogram;
end

%Train SVM
display ('Training SVM...');
SVM = svmtrain(double(TRAINING_LABELS), double(TRAINING_FEATURES), '-c 1 -g 0.07');

% Store support vector machine model
display ('Saving SVM...');
save(strcat(ROOT_DIR,'data/',POSITIVE_FILE, '_svm.dat'), 'SVM');

display ('Done.');