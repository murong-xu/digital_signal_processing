%% Task 2: Pyramid Representations
%% C2.1
im_in = im2double(imread('lenna_gray.jpg'));
gaussian_lp = [0,1,0;1,4,1;0,1,0] .* 1/8;
im_filtered = imfilter(im_in, gaussian_lp,'replicate');
im_downsampled = im_filtered(1:2:end,1:2:end);
im_upsampled = zeros(512,512);
for i = 1:256
    for j = 1:256
        im_upsampled(2*(i-1)+1,2*(j-1)+1) = im_downsampled(i,j);
    end
end
G = compute_reconstruction_filter();
im_result = imfilter(im_upsampled, G, 'replicate');
figure(1),imshow(im_in); title('original');
figure(2),imshow(im_filtered); title('LP-filtered');
figure(3),imshow(im_downsampled); title('downsampled');
figure(4),imshow(im_upsampled); title('upsampled(with zeros)');
figure(5),imshow(im_result); title('decovolutioned');

%% C2.2
im = im2double(imread('lenna_gray.jpg'));
F = 1/8 * [ 0 1 0 ; 1 4 1 ; 0 1 0 ]; % Gaussian filter
G = compute_reconstruction_filter(); % plug in here your values from C2.1 for reconstruction filter
laplacian_pyr = generate_laplacian_pyr(im, F, G);

figure,imshow(laplacian_pyr{1}); hold on; title('Level 1');
figure,imshow(laplacian_pyr{2}); hold on; title('Level 2');
figure,imshow(laplacian_pyr{3}); hold on; title('Level 3');
figure,imshow(laplacian_pyr{4}); hold on; title('Level 4');
figure,imshow(laplacian_pyr{5}); hold on; title('Level 5');

%% C2.3
% Analysis filter coefficients
f1d = [0.0267487574108098,-0.0168641184428750,-0.0782232665289879,0.266864118442872,0.602949018236358,0.266864118442872,-0.0782232665289879,-0.0168641184428750,0.0267487574108098]';

% Synthesis filter coefficients
g1d = [-0.0912717631142495, -0.0575435262284996, 0.591271763114247, 1.11508705245699, 0.591271763114247, -0.0575435262284996, -0.0912717631142495]';

% Compute the 2D filter kernels for analysis and synthesis CDF filters
[F,G] = compute_CDF_filter_kernel(f1d, g1d);

%% C2.4
im = im2double(imread('lenna_gray.jpg'));

% Compute Laplacian pyramid with Gaussian filter
F = 1/8 * [ 0 1 0 ; 1 4 1 ; 0 1 0 ]; % Gaussian filter
G = compute_reconstruction_filter(); % Gaussian reconstruction filter
laplacian_pyr_gauss = generate_laplacian_pyr(im, F, G);

% Compute Laplacian pyramid with CDF filter
% Analysis filter coefficients
f1d = [0.0267487574108098,-0.0168641184428750,-0.0782232665289879,0.266864118442872,0.602949018236358,0.266864118442872,-0.0782232665289879,-0.0168641184428750,0.0267487574108098]';
% Synthesis filter coefficients
g1d = [-0.0912717631142495, -0.0575435262284996, 0.591271763114247, 1.11508705245699, 0.591271763114247, -0.0575435262284996, -0.0912717631142495]';
% Compute the 2D filter kernels for analysis and synthesis CDF filters
[F,G] = compute_CDF_filter_kernel(f1d, g1d);
laplacian_pyr_CDF= generate_laplacian_pyr(im, F, G);
% Calculate Statistics
laplacian_pyr_CDF_var = calculate_statistics_pyramid(laplacian_pyr_CDF);
laplacian_pyr_gauss_var = calculate_statistics_pyramid(laplacian_pyr_gauss);

figure;
semilogy(laplacian_pyr_gauss_var, 'b.-'); hold on;
semilogy(laplacian_pyr_CDF_var, 'r.-');
title(sprintf('Variance (energy) in \n pyramid levels (log-scale)'));
legend('gauss', 'cdf'); xlabel('Level');

%% C2.5
im = im2double(imread('lenna_gray.jpg'));
figure,imshow(im); title('original');

% Laplacian pyramid with CDF filter
f1d = [0.0267487574108098,-0.0168641184428750,-0.0782232665289879,0.266864118442872,0.602949018236358,0.266864118442872,-0.0782232665289879,-0.0168641184428750,0.0267487574108098]';
% Synthesis filter coefficients
g1d = [-0.0912717631142495, -0.0575435262284996, 0.591271763114247, 1.11508705245699, 0.591271763114247, -0.0575435262284996, -0.0912717631142495]';
% Compute the 2D filter kernels for analysis and synthesis CDF filters
[F,G] = compute_CDF_filter_kernel(f1d, g1d);
laplacian_pyr_CDF= generate_laplacian_pyr(im, F, G);
[im_reconstruct_CDF, im_psnr_CDF] = reconstruct_image(laplacian_pyr_CDF, im, G);
figure,imshow(im_reconstruct_CDF); title('with CDF filter');


% Laplacian pyramid with Gaussian filter
F = 1/8 * [ 0 1 0 ; 1 4 1 ; 0 1 0 ]; % Gaussian filter
G = compute_reconstruction_filter(); % Gaussian reconstruction filter
laplacian_pyr_gauss = generate_laplacian_pyr(im, F, G);
[im_reconstruct_gauss, im_psnr_gauss] = reconstruct_image(laplacian_pyr_gauss, im, G);
figure,imshow(im_reconstruct_gauss); title('with Gaussian filter');

% A2.2

%% C2.1
function G = compute_reconstruction_filter()
% Output argument: G - filter kernel of size 3x3 of type double

% implement the filter here and save into G
G = 1/4 * [ 1 2 1 ; 2 4 2 ; 1 2 1 ]; % gaussian deconvolution kernel
end

%% C2.2
function laplacian_pyr = generate_laplacian_pyr(im, F, G)
% Input arguments: im - input image, of type double in range 0..1
% F - low pass filter
% G - reconstruction filter from C1.1 (3x3)
% Output arguments: laplacian_pyr - Laplacian pyramid (cell array with 5 elements, each corresponding to a particular level)

NL = 5; % number of levels

laplacian_pyr = cell(1,5); % allocate memory for the Laplacian pyramid
F = 1/8 * [ 0 1 0 ; 1 4 1 ; 0 1 0 ]; % Gaussian filter

im_in = im; %initialization step
% store the detailed info to level 1-4
for level=1:(NL-1)
    % Filter & downsample
    im_filtered = imfilter(im_in, F, 'replicate');
    im_downsampled = im_filtered(1:2:end, 1:2:end);
    % Upsample again & filter
    sz = size(im_downsampled)
    im_upsampled = zeros(sz*2);
    for i = 1:sz(1)
        for j = 1:sz(2)
            im_upsampled(2*(i-1)+1,2*(j-1)+1) = im_downsampled(i,j);
        end
    end
    im_result = imfilter(im_upsampled, G, 'replicate');
    % Compute difference image
    laplacian_pyr{level} = im_in - im_result;
    im_in = im_downsampled;
end
% store the lowest resolution image in level 5
laplacian_pyr{NL} = im_downsampled;
end
%% C2.3
function [F,G] = compute_CDF_filter_kernel(f1d, g1d)
% Input arguments: f1d - separable 1D version for analysis
% g1d - separable 1D version for synthesis
% Output arguments: F - 2D filter kernel for CDF wavelet for analysis
% G - 2D filter kernel for CDF wavelet for synthesis
F = f1d * f1d';
G = g1d * g1d';
end
%% C2.4
function var_pyramid = calculate_statistics_pyramid(pyramid_structure)
% Input arguments: pyramid_structure - computed pyramid for 5 levels (cell array, each element corresponds to a particular level)
% Output arguments: var_pyramid - variances over pyramid levels
NL = 5;
var_pyramid = zeros(NL,1); % allocate memory
b = 4;              % ignore margins
for i = 1: NL
    current_level = pyramid_structure{i};
    current_level = current_level(1+b: end-b, 1+b: end-b); % remove borders
    var_pyramid(i) = var(current_level(:));
end
end

%% C2.5
function [im_reconstruct, im_psnr] = reconstruct_image(pyramid, im, G)
% Input arguments: pyramid - pyramid representation of the image (cell array with 5 levels)
% im - input image, of type double in range 0..1
% G - reconstruction filter
% Output arguments: im_reconstruct - reconstructed image
% im_psnr - psnr of the difference image

NL = numel(pyramid);
b = 4; % border to ignore for PSNR

% 1. first set two levels to 0 (i.e. elements 1 and 2)
pyramid{1} = zeros(size(pyramid{1}));
pyramid{2} = zeros(size(pyramid{2}));

im_in = pyramid{NL};
% 2. Reconstruct images:
for level = NL-1:-1:1
    % 2.a Upscale & filter previous (coarser) level
    im_upsampled = zeros(size(pyramid{level}));
    sz = size(im_in)
    for i = 1:sz(1)
        for j = 1:sz(2)
            im_upsampled(2*(i-1)+1,2*(j-1)+1) = im_in(i,j);
        end
    end
    im_filtered = imfilter(im_upsampled, G, 'replicate');
    % 2.b Add current level (difference image)
    im_reconstruct = im_filtered + pyramid{level};
    im_in = im_reconstruct;
end
% 3. Compute difference image and PSNR
% im_diff = im - im_reconstruct;
% im_diff = im_diff(1+b: end-b, 1+b: end-b); % remove borders
% im_diff_quadrat = im_diff.^2;
% mse = mean(im_diff_quadrat(:));
% im_psnr = 10 * log10 (1/ mse );
im_reconstruct_small = im_reconstruct(1+b: end-b, 1+b: end-b); % remove borders
im_small = im(1+b: end-b, 1+b: end-b);
im_psnr = psnr(im_reconstruct_small, im_small);

end

%% A2.2