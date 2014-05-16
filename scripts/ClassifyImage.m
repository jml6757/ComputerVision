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
SVM = load(strcat(ROOT_DIR,'data/', 'category', int2str(CATEGORY), '_svm.dat'),'-mat');
SVM = SVM.SVM;

% Allocate structures for SVM arguments
% TRAINING_LABELS = zeros(length(DATA), 1);
TRAINING_FEATURES = [];
COUNT = 1;

%Add data to structured SVM training arrays
display ('Formatting Data...');
for i = 1:length(DATA)
    if(strcmp(DATA(i).train_test,'test'))
        if(DATA(i).category == CATEGORY)
            LABEL = 1;
        else
            LABEL = -1;
        end
        TEST_LABELS(COUNT,1) =  LABEL;
        TRAINING_FEATURES = [TRAINING_FEATURES; DATA(i).histogram];
        COUNT = COUNT + 1;
    end
end

%Train SVM
display ('Predicting with SVM...');

PREDICT_LABELS = svmpredict(TEST_LABELS, TRAINING_FEATURES, SVM, '-b 1');

display ('Done.');

LESS_THAN = bsxfun(@lt, TEST_LABELS, PREDICT_LABELS);
GREATER_THAN = bsxfun(@gt, TEST_LABELS, PREDICT_LABELS);
FALSE_POSITIVE = sum(LESS_THAN)/(COUNT - 1)*100
FALSE_NEGATIVE = sum(GREATER_THAN)/(COUNT - 1)*100

