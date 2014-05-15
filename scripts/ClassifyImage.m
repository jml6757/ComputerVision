%TrainSVM - Opens the extracted feature sets from data files
%           and trains a support vector machine for the chosen
%           data set.


% Parameters
CATEGORY = 1; % Train with this category as the positive images

% Get the project directory and set up paths
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/ClassifyImage','');
DATA_PATH = strcat(ROOT_DIR, 'data/');
LIB_PATH  = strcat(ROOT_DIR, 'lib/');
addpath(DATA_PATH);
addpath(LIB_PATH);

% Load data
display ('Loading Data...');
DATA = load(strcat(DATA_PATH, 'image_data.dat'),'-mat');
DATA = DATA.DATA;

% Load SVM
SVM = load(strcat(DATA_PATH, 'category1_svm.dat'),'-mat');
SVM = SVM.SVM;

% Allocate structures for SVM arguments
TRAINING_LABELS = zeros(length(DATA), 1);
TRAINING_FEATURES = [];

%Add data to structured SVM training arrays
display ('Formatting Data...');
for i = 1:length(DATA)
    if(DATA(i).category == CATEGORY)
        LABEL = 1;
    else
        LABEL = -1;
    end
    TRAINING_FEATURES = [TRAINING_FEATURES; DATA(i).histogram];
end

%Train SVM
display ('Predicting with SVM...');
[predict_label, accuracy, prob_estimates] = svmpredict(TRAINING_LABELS, TRAINING_FEATURES, SVM, '-b 1');



