function histogram = getHist(features, C)

% Get all the dinstances to the centroids
for i = 1:size(features,1)
    for j = 1:size(C,1)
        dist(i,j) = norm(features(i,:) - C(j,:));
    end
    [minimum,index(i)] = min(dist(i,:));
end
histogram = hist(index, 512);
 
end