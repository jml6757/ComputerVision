%TrainSVM - Opens the extracted feature sets from data files
%           and trains a support vector machine for the chosen
%           data set.

% Input and output arguments
POSITIVE_FILE = 'airplanes_side';
NEGATIVE_FILE = ['background     '; 'cars_brad      '; 'cars_markus    '; 'faces          '; 'leaves         '; 'motorbikes_side'];
NEGATIVE_FILE = cellstr(NEGATIVE_FILE);

% Get the project directory
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/TrainSVM','');

% Set the data/libary location and add to the global matlab path
DATA_PATH = strcat(ROOT_DIR, 'data/');
LIB_PATH  = strcat(ROOT_DIR, 'lib/');
addpath(DATA_PATH);
addpath(LIB_PATH);

% Number of images
NUM_POS = 50;                     % Number of positive instances
NUM_NEG = 50;                     % Number of negative instances
NUM_TRAINING = NUM_NEG + NUM_POS; % Total number of training images
NUM_FEATURES = 128;               % Number of features per image

% Preallocate arrays for SVM training
% TRAINING_LABELS   = zeros(NUM_TRAINING,1);            % Positive or negative label
% TRAINING_FEATURES = zeros(NUM_TRAINING,NUM_FEATURES); % Features for all images

%Load positive data
POS_DATA = load(strcat(DATA_PATH, POSITIVE_FILE, '.dat'),'-mat');
POS_DATA = POS_DATA.FEATURES;

%Load negative data
for i = 1:length(NEGATIVE_FILE)
    NEG_DATA(i) = load(strcat(DATA_PATH, strtrim(char(NEGATIVE_FILE(i))), '.dat'),'-mat');
end

%Shuffle data


%Add data to structured SVM training arrays
POS = 0;
for i = 1:NUM_POS
    TRAINING_LABELS(i) = 1;
end

for i = 1:NUM_NEG
    TRAINING_LABELS(i+NUM_POS) = -1;
end


%Train SVM
%SVM = svmtrain(TRAINING_LABELS, TRAINING_FEATURES, '-c 1 -g 0.07');
        
% Store support vector machine model
%save(strcat(ROOT_DIR,'data/',POSITIVE_FILE, '_svm.dat'), 'SVM');