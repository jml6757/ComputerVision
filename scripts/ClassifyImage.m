%ClassifyImage - Classifies specified images

% Input and output arguments
POSITIVE_FILE = 'airplanes_side';
NEGATIVE_FILE = ['background     '; 'cars_brad      '; 'cars_markus    '; 'faces          '; 'leaves         '; 'motorbikes_side'];
NEGATIVE_FILE = cellstr(NEGATIVE_FILE);

% Get the project directory
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/ClassifyImage','');

% Set the data/libary location and add to the global matlab path
DATA_PATH = strcat(ROOT_DIR, 'data/');
LIB_PATH  = strcat(ROOT_DIR, 'lib/');
addpath(DATA_PATH);
addpath(LIB_PATH);

%Load positive data
POS_DATA = load(strcat(DATA_PATH, POSITIVE_FILE, '.dat'),'-mat');
POS_DATA = POS_DATA.FEATURES;

%Load negative data
% for i = 1:length(NEGATIVE_FILE)
%     NEG_DATA(i) = load(strcat(DATA_PATH, strtrim(char(NEGATIVE_FILE(i))), '.dat'),'-mat');
% end

%Load SVM
SVM = load(strcat(ROOT_DIR,'data/',POSITIVE_FILE, '_svm.dat'),'-mat');
SVM = SVM.SVM;

% NEG_SET = NEG_DATA(1).FEATURES;
% NEG_SET = NEG_SET(3)
% [NUM_FEATS, NUM_POINTS] = size(NEG_SET.surfFeatures);
% SUM = 0;
% for j = 1:NUM_FEATS
%     SUM = SUM + svmpredict(0, double(NEG_SET.surfFeatures(j,:)), SVM);
% end

NUM_POS = 50;
POS = 0;
NEG = 0;
for i = 1:NUM_POS
    SUM = 0;
    
    for j = 1:length(POS_DATA(i).surfFeatures)
        SUM = SUM + svmpredict(0, double(POS_DATA(i).surfFeatures(j,:)), SVM);
    end
    
    if(SUM > 0)
       POS = POS + 1; 
    else
       NEG = NEG + 1;
    end
end

POS
NEG

