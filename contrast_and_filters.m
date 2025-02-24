% This is for the final 2 questions of quiz 1
%
% This script demonstrates:
%   1) Piecewise Linear Contrast Stretching on "einstein.tif" and "race.tif"
%   2) Weighted Averaging, Laplacian, and Sobel filtering on "race.tif"
%
% Make sure "einstein.tif" and "race.tif" are in the current folder before running
% Adjust the threshold pairs (T1, T2) if needed.

function contrast_and_filters()
    % PART A: CONTRAST STRETCHING
    % ------------------------------------
    I_einstein = imread('einstein.tif');  % 8-bit grayscale
    I_race     = imread('race.tif');      % also 8-bit grayscale

    % Try three different (T1, T2) pairs
    T1_values = [50,  70,  90 ];
    T2_values = [150, 180, 200];

    % Einstein
    for k = 1:length(T1_values)
        T1 = T1_values(k);
        T2 = T2_values(k);
        stretchedImg = stretch(I_einstein, T1, T2);
        outName = sprintf('einstein_stretch_%d_%d.tif', T1, T2);
        imwrite(stretchedImg, outName);
        fprintf('Wrote %s\n', outName);
    end

    % Race
    for k = 1:length(T1_values)
        T1 = T1_values(k);
        T2 = T2_values(k);
        stretchedImg = stretch(I_race, T1, T2);
        outName = sprintf('race_stretch_%d_%d.tif', T1, T2);
        imwrite(stretchedImg, outName);
        fprintf('Wrote %s\n', outName);
    end


    % PART B: SPATIAL FILTERING ON "race.tif"
    % ------------------------------------
    race_double = im2double(I_race);  % convert to double

    % (1) Weighted averaging (3x3)
    h_avg = (1/16) * [1 2 1; 2 4 2; 1 2 1];
    race_smoothed = imfilter(race_double, h_avg, 'replicate');
    imwrite(im2uint8(race_smoothed), 'race_smoothed.tif');
    fprintf('Wrote race_smoothed.tif\n');

    % (2) Laplacian filter
    lap_mask = [0 -1 0; -1 4 -1; 0 -1 0];
    race_lap = imfilter(race_double, lap_mask, 'replicate');

    % Sharpening: f_sharp = f - lambda*(f * Laplacian)
    lambda = 1.0;
    race_sharp = race_double - lambda * race_lap;
    % Clip to [0,1] to avoid negative or >1
    race_sharp = max(min(race_sharp,1),0);
    imwrite(im2uint8(race_sharp), 'race_sharp.tif');
    fprintf('Wrote race_sharp.tif\n');

    % (3) Sobel x- and y- filters
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];

    race_sx = imfilter(race_double, sobel_x, 'replicate');
    race_sy = imfilter(race_double, sobel_y, 'replicate');

    % For quick visualization, shift and scale to [0..1] or [0..255]:
    race_sx_uint8 = im2uint8( (race_sx + 1)/2 );  % shift to range [0,1]
    race_sy_uint8 = im2uint8( (race_sy + 1)/2 );
    imwrite(race_sx_uint8, 'race_sobel_x.tif');
    imwrite(race_sy_uint8, 'race_sobel_y.tif');
    fprintf('Wrote race_sobel_x.tif and race_sobel_y.tif\n');

    % Gradient magnitude
    grad_mag = sqrt(race_sx.^2 + race_sy.^2);
    grad_mag = grad_mag ./ max(grad_mag(:));  % normalize to [0,1]
    imwrite(im2uint8(grad_mag), 'race_sobel_magnitude.tif');
    fprintf('Wrote race_sobel_magnitude.tif\n');
end


% =====================
%  Piecewise Stretch
% =====================
function Y = stretch(X, T1, T2)
    % X  : input image (grayscale, 0..255)
    % T1 : lower threshold
    % T2 : upper threshold
    % Y  : output image after piecewise linear transformation
    %
    % The transform is:
    %   if X < T1 => 0
    %   if T1 <= X <= T2 => 255*(X - T1)/(T2 - T1)
    %   if X > T2 => 255

    Xdouble = double(X);

    Y = zeros(size(Xdouble));

    % Region 1: X < T1
    mask1 = (Xdouble < T1);
    Y(mask1) = 0;

    % Region 2: T1 <= X <= T2
    mask2 = (Xdouble >= T1 & Xdouble <= T2);
    Y(mask2) = 255 .* (Xdouble(mask2) - T1) ./ (T2 - T1);

    % Region 3: X > T2
    mask3 = (Xdouble > T2);
    Y(mask3) = 255;

    Y = uint8(Y);
end
