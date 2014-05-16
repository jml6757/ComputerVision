function histogram = getHist(features, C)

% Get all the dinstances to the centroids
for i = 1:size(features,1)
    for j = 1:size(C,1)
        dist(i,j) = norm(features(i,:) - C(j,:));
    end
    [minimum,index(i)] = min(dist(i,:));
end
histogram = hist(index, 512);

%     % Get all the distances to the centroids
%     for i = 1:size(features,1)
%         for j = 1:size(C,1)
%             dist(i,j) = norm(features(i,:) - C(j,:));
%         end
%     end
%     
%     % Assign features to centroids
%     for i = 1:size(features,1)
%        [x,I] = min(dist(i,:),[],1);
%     end
%     
%     % Build the histogram
%     hist = zeros(size(C,1),1);
%     for i = 1:size(I,1)
%         hist(I(i)) = hist(I(i)) + 1;
%     end
%     hist = hist'; % dont mind me :)
    
end