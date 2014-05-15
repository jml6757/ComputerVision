% Extracts Features from all images

% Image folders
FOLDERS = ['airplanes_side '; 'background     '; 'cars_brad      '; 'cars_markus    '; 'faces          '; 'leaves         '; 'motorbikes_side'];
FOLDERS = cellstr(FOLDERS);

% Get the project directory
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/BuildClusters','');

% Structure to store all file data
DATA = struct('category', 0, 'directory',  0, 'filename',  0, 'numFeatures', 0, 'surfFeatures', 0, 'histogram', 0);

% Fetch all file names
display ('Getting Filenames...');
COUNT = 1;
for i = 1:length(FOLDERS)
    IMAGE_PATH = strcat(ROOT_DIR, 'images/', strtrim(char(FOLDERS(i))), '/');
    FILE_DESCRIPTORS = dir(strcat(IMAGE_PATH, '*.jpg'));
    for j = 1:length(FILE_DESCRIPTORS)
        DATA(COUNT).category = i;
        DATA(COUNT).directory = IMAGE_PATH;
        DATA(COUNT).filename = FILE_DESCRIPTORS(j).name;
        COUNT = COUNT + 1;
    end
end

display ('Extracting Features...');

X = [];

% Get features from all images
for i = 1:length(DATA)
    
    %Display iteration
    display(strcat(DATA(i).directory,  DATA(i).filename));
    
    %Load image
    IMAGE = imread(strcat(DATA(i).directory,  DATA(i).filename));
    
    %Convert to grayscales if necessary
    if (size(IMAGE, 3) == 3)
        IMAGE = rgb2gray(IMAGE);
    end;
    
    %Store extracted features to vector
    [Y, Z] = extractFeatures(IMAGE, detectSURFFeatures(IMAGE));
    DATA(i).surfFeatures = Y;
    [FEATS, DIMS] = size(Y);
    DATA(i).numFeatures = FEATS;
    X = [X; Y];
end

display ('Saving Data...');

save(strcat(ROOT_DIR,'data/','image_data.dat'), 'DATA');
save(strcat(ROOT_DIR,'data/','feature_array.dat'), 'X');

display ('Done.');



