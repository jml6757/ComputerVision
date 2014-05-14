function HogDescriptor = HogDes(I, CellSize, NoBins)

I=single(I)/255;

% Compute Gradients
[Gx,Gy]=gradient(I); 
Magnitude = sqrt(Gx.^2+Gy.^2);
Orientations = atan2(Gy,Gx) * (180/pi) + (180.*ones(size(I)));

[Height, Width] = size(I);

% Find number of cells in horizontal and vertical direction
HorzCells = ceil(Width/CellSize);
VertCells = ceil(Height/CellSize);

% Histogram for each cell.
Hist = zeros(VertCells, HorzCells, NoBins);

% ************* Cell Histograms *************
for row = 1:VertCells

    % Find the Row pixel number for current Cell.
    StartRow = ((row-1)*CellSize) + 1;
    
    if(StartRow + CellSize - 1> Height)
        EndRow = Height;
    else
        EndRow = StartRow + CellSize - 1;
    end

    for col = 1: HorzCells
        % Find the Row pixel number for current Cell.
        
        StartCol = ((col-1)*CellSize) + 1;
        
        if(StartCol + CellSize - 1> Width)
            EndCol = Width;
        else
            EndCol = StartCol + CellSize - 1;
        end

        % Get the Magnitude and Angles for this cell
        CellMag = Magnitude (StartRow : EndRow, StartCol : EndCol);
        CellOrient = Orientations (StartRow : EndRow, StartCol : EndCol);
        
        % Compute Histogram for each cell
        Hist(row, col, :) = GetHistogram(CellMag(:), CellOrient(:), NoBins);        
    end
end

% ************* Block Normalization *************
HogDescriptor = [];

% Take 2x2 blocks of cells and normalize the histograms.
for row = 1: (VertCells-1)       
    for col = 1: (HorzCells-1)
        % Get histograms for the cells in this block.
        BlockHists = Hist (row:row+1, col:col+1, :);
        Magnit = norm (BlockHists(:)) + 0.01;   % Compute the magnitude.
        Normalized = BlockHists/Magnit; % Normilaize.
        HogDescriptor = [HogDescriptor; Normalized(:)]; % Add to the Descriptor
    end
end

HogDescriptor = reshape(HogDescriptor, [floor(Height/CellSize), floor(Width/CellSize), NoBins*4]);
