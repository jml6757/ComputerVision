I = imread('..\images\airplanes_side\0001.jpg', 'jpg');
I = rgb2gray(I);
[featureVector, hogVisualization] = extractHOGFeatures(I);
%imshow(I)