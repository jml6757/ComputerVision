%GetFeatures - Opens an set of images and extracts the features.
%              This data is saved the folder name.


% Get the project directory
ROOT_DIR = strrep(mfilename('fullpath') ,'scripts\GetFeatures','');

% Set the path location and add to the global matlab path
IMAGE_PATH = strcat(ROOT_DIR, 'images\airplanes_side\');
addpath(IMAGE_PATH);

% Get all file descriptors in the path
FILES = dir(strcat(IMAGE_PATH, '*.jpg'));
NUM_FILES = length(FILES);

for i = 1:NUM_FILES
   IMAGE = imread(FILES(i).name);
end

imshow(IMAGE)