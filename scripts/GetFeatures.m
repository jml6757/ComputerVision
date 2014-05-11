%GetFeatures - Opens an set of images and extracts the features.
%              This data is saved the folder name.

% Input and output arguments
INPUT_FOLDER = 'airplanes_side'

% Get the project directory
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/GetFeatures','');

% Set the path location and add to the global matlab path
IMAGE_PATH = strcat(ROOT_DIR, 'images/', INPUT_FOLDER, '/');
addpath(IMAGE_PATH);

% Get all image file descriptors in the path
FILE_DESCRIPTORS = dir(strcat(IMAGE_PATH, '*.jpg'));
NUM_FILES = length(FILE_DESCRIPTORS);

% Allocate array to hold extracted features
FEATURES = struct('filename',  0, 'surf', 0, 'fast', 0, 'brisk', 0);

for i = 1:NUM_FILES
    
    %Display iteration
    display(strcat('Image: ', FILE_DESCRIPTORS(i).name));
    
    %Load image
    IMAGE = rgb2gray(imread(FILE_DESCRIPTORS(i).name));
    
    %Store extracted features to vector
    FEATURES(i).filename = FILE_DESCRIPTORS(i).name;
    %FEATURES(i).surf     = detectSURFFeatures(IMAGE);
    %FEATURES(i).fast     = detectFASTFeatures(IMAGE);
    %FEATURES(i).brisk    = detectBRISKFeatures(IMAGE);
end

% Store extracted feature vector to disk
save(strcat(ROOT_DIR,'data/',INPUT_FOLDER, '.dat'), 'FEATURES');
