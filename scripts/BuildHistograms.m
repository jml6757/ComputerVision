% Build histograms from IDX data and image DATA

% Setup Paths
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/BuildHistograms','');
DATA_PATH = strcat(ROOT_DIR, 'data/');

% Load Saved Data
display('Loading Data...');
IDX = load(strcat(DATA_PATH, 'cluster_index.dat'),'-mat');
IDX = IDX.IDX;
DATA = load(strcat(DATA_PATH, 'image_data.dat'),'-mat');
DATA = DATA.DATA;

% Build Histograms
display('Building Histograms...');

START = 1;
for i = 1:length(DATA)
    NUM_FEATS = DATA(i).numFeatures;
    END = START + NUM_FEATS;
    DATA(i).histogram = hist(IDX(START:END), 256);
    START = END;
end

% Overwrite Old Data File
display('Saving Data...');
save(strcat(ROOT_DIR,'data/','image_data.dat'), 'DATA');

display('Done.');