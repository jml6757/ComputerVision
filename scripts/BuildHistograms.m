% Build histograms from IDX data and image DATA
HIST_SIZE = 512;

% Setup Paths
ROOT_DIR = strrep(strrep(mfilename('fullpath'), '\', '/') ,'scripts/BuildHistograms','');
DATA_PATH = strcat(ROOT_DIR, 'data/');

% Load Saved Data
display('Loading Data...');
IDX = load(strcat(DATA_PATH, 'cluster_index.dat'),'-mat');
IDX = IDX.IDX;
DATA = load(strcat(DATA_PATH, 'image_data.dat'),'-mat');
DATA = DATA.DATA;
C = load(strcat(DATA_PATH, 'clusters.dat'),'-mat');

% Build Histograms
display('Building Histograms...');

START = 1;
for i = 1:length(DATA)
    if strcmp(DATA(i).train_test,'train')
        NUM_FEATS = DATA(i).numFeatures;
        END = START + NUM_FEATS-1;
        HIST = hist(IDX(START:END), HIST_SIZE);
        DATA(i).histogram = HIST/norm(HIST);
        START = END;
    else
        DATA(i).histogram = getHist(DATA(i).surfFeatures,C);
    end
end

% Overwrite Old Data File
display('Saving Data...');
save(strcat(ROOT_DIR,'data/','image_data.dat'), 'DATA');

display('Done.');