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
FEATURES = struct('filename',  0, 'surfFeatures', 0, 'freakFeatures', 0, 'surfPoints', 0, 'freakPoints', 0);

for i = 1:NUM_FILES
    
    %Display iteration
    display(strcat('Image: ', FILE_DESCRIPTORS(i).name));
    
    %Load image
    IMAGE = imread(FILE_DESCRIPTORS(i).name);
    if (size(IMAGE, 3) == 3)
        IMAGE = rgb2gray(IMAGE);
    end;
    
    %Store extracted features to vector
    FEATURES(i).filename = FILE_DESCRIPTORS(i).name;
    tmp = detectSURFFeatures(IMAGE);
    [FEATURES(i).surfFeatures, FEATURES(i).surfPoints]     = extractFeatures(IMAGE, tmp.selectStrongest(100));
    tmp = detectFASTFeatures(IMAGE);
    [FEATURES(i).freakFeatures, FEATURES(i).freakPoints]   = extractFeatures(IMAGE, tmp.selectStrongest(100), 'Method', 'FREAK');
    clear tmp;
end

% Store extracted feature vector to disk
save(strcat(ROOT_DIR,'data/',INPUT_FOLDER, '.dat'), 'FEATURES');
