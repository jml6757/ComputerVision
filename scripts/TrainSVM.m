%TrainSVM - Opens the extracted feature sets from data files
%           and trains a support vector machine for the chosen
%           data set.

% Input and output arguments
POSITIVE_FILE = 'airplanes_side';
NEGATIVE_FILE = ['background', 'cars', 'faces', 'leaves', 'motorbikes_side'];

% Get the project directory
ROOT_DIR = strrep(mfilename('fullpath') ,'scripts\TrainSVM','');

% Set the data location and add to the global matlab path
DATA_PATH = strcat(ROOT_DIR, 'data\');
addpath(DATA_PATH);

% Number of images
NUM_POS = 50;                     % Number of positive instances
NUM_NEG = 50;                     % Number of negative instances
NUM_TRAINING = NUM_NEG + NUM_POS; % Total number of training images
NUM_FEATURES = 128;               % Number of features per image

% Preallocate arrays for SVM training
TRAINING_LABELS   = zeros(NUM_TRAINING,1);
TRAINING_FEATURES = zeros(NUM_TRAINING,NUM_FEATURES);

%Load and format data
POS_DATA = load(strcat(DATA_PATH, POSITIVE_FILE, '.dat'),'-mat');
POS_DATA = POS_DATA.FEATURES;


%Train SVM

% svmtrain(training_label_vector, training_instance_matrix [, 'libsvm_options']);
% -training_label_vector:
%     An m by 1 vector of training labels (type must be double).
% -training_instance_matrix:
%     An m by n matrix of m training instances with n features.
%     It can be dense or sparse (type must be double).
% -libsvm_options:
%     A string of training options in the same format as that of LIBSVM.
%SVM = svmtrain(LABELS, FEATURES, '-c 1 -g 0.07');
        
