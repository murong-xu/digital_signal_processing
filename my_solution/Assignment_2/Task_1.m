%% Task 1: Image Sampling
%% C1.1
N = 400;
sz = 512;
[gridX, gridY] = generate_rect_grid(sz, N);
figure,scatter(gridX(:), gridY(:)); grid on; box on; xlim([1 10]); ylim([1 10]);

%% C1.2
sz = 512;
N = 400;
[gridX, gridY] = generate_rect_grid(sz, N);
im = im2double(imread('lenna_gray.jpg'));
[im_linear, im_cubic] = interp_im_rect_grid(im, gridX, gridY);
figure,imshow(im_linear); %1
figure,imshow(im_cubic); % 2
figure,imshow(im); %3

%% C1.3
Load the original image
im = im2double(imread('lenna_gray.jpg'));
sz = min(size(im));

% Subsample the image with rectangular grid and interpolate to grid locations
N = 400;
[gridX, gridY] = generate_rect_grid(sz, N);
[im_small_linear, im_small_cubic] = interp_im_rect_grid(im, gridX, gridY);

% Interpolate the image to its original size
im_rec = reconstruct_from_smaller_image(im_small_linear, sz, gridX, gridY);
figure,imshow(im);%1
figure,imshow(im_small_linear);%2
figure,imshow(im_rec);%3

%% C1.4
[hexaX, hexaY] = generate_hexagonal_grid();
figure,scatter(hexaX(:), hexaY(:)); grid on; box on; xlim([1 10]); ylim([1 10]);

%% C1.5
im = im2double(imread('lenna_gray.jpg'));
N = 400;
sz = size(im);

[hexaX, hexaY] = generate_hexagonal_grid();
[im_rec] = interpolate_reconstruct(im, hexaX, hexaY);
figure, imshow(im_rec(200:300,200:300));
figure,imshow(im(200:300,200:300))

%% C1.6
im1 = im2double(imread('lenna_gray.jpg'));
im2 = im1;
im2(1:50,1:50) = 0;
psnr_diff = compute_psnr_diff(im1, im2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSNR comparison between hexagonal and rectangular sampling
im = im2double(imread('lenna_gray.jpg'));
sz = size(im);
N = 400;
% Rectangular sampling
[gridX, gridY] = generate_rect_grid(sz, N);
[im_linear, im1_cubic] = interp_im_rect_grid(im, gridX, gridY);
im_recon_rect = reconstruct_from_smaller_image(im_linear, sz, gridX, gridY);

% Hexagonal sampling
[hexaX, hexaY] = generate_hexagonal_grid();
im_recon_hexa = interpolate_reconstruct(im, hexaX, hexaY);

% PSNR
psnr_hexa = compute_psnr_diff(im_recon_hexa, im)
psnr_rect = compute_psnr_diff(im_recon_rect, im)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% C1.1
function [gridX, gridY] = generate_rect_grid(sz, N)
% Input arguments: sz - size of the input image (square shape is assumed)
% N - required side size of the grid (so that we have a grid of size NxN)
% Output argument: [gridX, gridY] - X and Y coordinates of the rectangular grid
% implement grid calculation here
sz = min(sz);
sampled_rectangular = linspace(1, sz, N);
[gridX, gridY] = meshgrid(sampled_rectangular);
end

%% C1.2
function [im_linear, im_cubic] = interp_im_rect_grid(im, gridX, gridY)
% Input arguments: im - input image to interpolate, of type double in range 0..1
% gridX - grid X coordinates to which to interpolate
% gridY - grid Y coordinates to which to interpolate
% Output arguments: im_linear - image of linear interpolation
% im_cubic - image of cubic interpolation

sz = min(size(im));
[imX, imY] = meshgrid(1:sz);   % original image is given at these positions

% now implement the interpolation using different options: cubic, linear
im_linear = interp2(imX, imY, im, gridX, gridY, 'linear');
im_cubic = interp2(imX, imY, im, gridX, gridY, 'cubic');
end

%% C1.3
function im_rec = reconstruct_from_smaller_image(im_small, sz, gridX, gridY)
% Input arguments:
% im_small - image of smaller size of type double in range (0,1),
% from which reconstruction to be performed
% sz - size of the original image (square image assumed)
% gridX - grid X coordinates to which to interpolated
% gridY - grid Y coordinates to which to interpolated
% Output arguments: im_rec - reconstructed image

N = size(gridX, 1);
sz = min(sz);

% 1. Apply triscatteredinterp function
[rec_X, rec_Y] = meshgrid(1:sz);
F = TriScatteredInterp(gridX(:), gridY(:), im_small(:), 'linear'); % input of TriScatteredInterp must be column vector

% 2. Then interpolate image to the original image coordinates (these have to be computed first)
im_rec = F(rec_X, rec_Y);
end

%% C1.4
function [hexaX, hexaY] = generate_hexagonal_grid()
% Output arguments:
% hexaX - X coordinates of the hexagonal grid (matrix)
% hexaY - Y coordinates of the hexagonal grid (matrix)
sz = 512;
dx=(511-1)/(372-1);    % horizontal displacement in hexaX
dy = dx * sqrt(3)/2;   % vertical displacement in hexaY

% Compute hexaX and hexaY using meshgrid()
sampled_hexa_x = 1: dx: 511+ 0.5 * dx;
sampled_hexa_y = 1: dy: 512;
[gridX, gridY] = meshgrid(sampled_hexa_x, sampled_hexa_y);

% Apply a shift on each 2nd raw by 0.5*dx to obtain the sampling scheme depicted in figure
hexaX = gridX;
hexaX(2:2:end, :) = gridX(2:2:end, :) + 0.5 * dx;
hexaY = gridY;
end

%% C1.5
function [im_rec] = interpolate_reconstruct(im, hexaX, hexaY)
% Input arguments: im - input grayscale image, of type double, in range 0..1
% hexaX, hexaY - hexagonal grid coordinates for X and Y
% Output arguments: im_rec - reconstructed grayscale image, of type double in range 0..1
sz = min(size(im));
% use linear interpolation
[rec_X, rec_Y] = meshgrid(1:sz);
im_linear = interp2(rec_X, rec_Y, im, hexaX, hexaY, 'linear');
F = TriScatteredInterp(hexaX(:), hexaY(:), im_linear(:), 'linear');
im_rec = F(rec_X, rec_Y);
end

%% C1.6
function psnr_diff = compute_psnr_diff(im, im_rec)
% Input arguments: im - original grayscale image, of type double, in range 0..1
% im_rec -reconstructed grayscale image, of type double, in range 0..1
% Output arguments: psnr_diff - psnr of the difference image (im_rec-im)
border = 5;
% implement psnr computation according to the formula
im = im(border+1 : end-border, border+1 : end-border);
im_rec = im_rec(border+1 : end-border, border+1 : end-border);
mse = immse(im, im_rec);
psnr_diff = 10 * log10(1./mse); % notice that log() is ln()
end