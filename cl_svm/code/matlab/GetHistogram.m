function H = GetHistogram(magnitudes, angles, numBins)

max_angle = 180;

% Compute the Bin Size 
binSize = max_angle/numBins;
bin = 0;
H = zeros (1,numBins);

% Compute the histogram
for i = 0: binSize: (max_angle-10)
    bin = bin+1;
    for k = 1: numel(angles)
        if (angles(k)>=i && angles(k) < i+binSize)
            H (bin) = H(bin) + magnitudes(k);
        end
    end
end

H(1) = H(1) + sum (magnitudes(find(angles >=max_angle)));
end