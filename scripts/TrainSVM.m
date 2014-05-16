%TrainSVM - Opens the extracted feature sets from data files
%           and trains a support vector machine for the chosen
%           data set.

% Parameters
CATEGORY = 6; % Train with this category as the positive images

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

% Allocate structures for SVM arguments
% TRAINING_LABELS = zeros(length(DATA), 1);
TRAINING_FEATURES = [];
COUNT = 1;

%Add data to structured SVM training arrays
display ('Formatting Data...');
for i = 1:length(DATA)
    if(strcmp(DATA(i).train_test,'train'))
        if(DATA(i).category == CATEGORY)
            LABEL = 1;
        else
            LABEL = -1;
        end
        TRAINING_LABELS(COUNT,1) =  LABEL;
        TRAINING_FEATURES = [TRAINING_FEATURES; DATA(i).histogram];
        COUNT = COUNT + 1;
    end
end

%Train SVM
display ('Training SVM...');
SVM = svmtrain(double(TRAINING_LABELS), double(TRAINING_FEATURES), '-c 1 -g 0.07 -b 1');

% Store support vector machine model
display ('Saving SVM...');
save(strcat(ROOT_DIR,'data/', 'category', int2str(CATEGORY), '_svm.dat'), 'SVM');

display ('Done.');