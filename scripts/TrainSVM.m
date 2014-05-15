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
NUM_FEATURES = 64;                % Number of features per image

display ('Loading Data...');

%Load positive data
POS_DATA = load(strcat(DATA_PATH, POSITIVE_FILE, '.dat'),'-mat');
POS_DATA = POS_DATA.FEATURES;

%Load negative data
for i = 1:length(NEGATIVE_FILE)
    NEG_DATA(i) = load(strcat(DATA_PATH, strtrim(char(NEGATIVE_FILE(i))), '.dat'),'-mat');
end

display ('Formatting Data...');

%Shuffle data
TRAINING_LABELS = [];
TRAINING_FEATURES = [];

%Add data to structured SVM training arrays
for i = 1:NUM_POS
    for j = 1:length(POS_DATA(i).surfFeatures)
        TRAINING_LABELS = [TRAINING_LABELS 1];
        TRAINING_FEATURES = [TRAINING_FEATURES POS_DATA(i).surfFeatures(j,:)'];
    end
end

for k = 1:6
    NEG_SET = NEG_DATA(k).FEATURES;
    for i = 1:10
        [NUM_FEATS, NUM_POINTS] = size(NEG_SET(i).surfFeatures);
        for j = 1:NUM_FEATS
            TRAINING_LABELS = [TRAINING_LABELS -1];
            TRAINING_FEATURES = [TRAINING_FEATURES NEG_SET(i).surfFeatures(j,:)'];
        end
    end
end

display ('Training SVM...');

%Train SVM
SVM = svmtrain(double(TRAINING_LABELS'), double(TRAINING_FEATURES'), '-c 1 -g 0.07');

display ('Saving SVM...');

% Store support vector machine model
save(strcat(ROOT_DIR,'data/',POSITIVE_FILE, '_svm.dat'), 'SVM');

display ('Done.');