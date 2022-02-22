%% Task 2: Separable Block Transforms
%% C2.1
image_in = im2double(imread('pears.jpg'));
% Function to be applied on each block: block_func
block_func = @(block, block_arg) block+block_arg; % increasing brightness
block_arg = 0.4;
block_size = 16;
% Block splitter: split the image with the given block sizes into blocks and apply block_func on each
image_out = block_splitter_image(image_in, block_size, block_func, block_arg);

figure
imshow(image_out);
title('Block transformed image');
hold off
figure
imshow(image_in);
title('Input image');

%% C2.2
im = im2double(imread('pears.jpg'));
block_in = im(1:250,1:250);
block_out = block_processor(block_in);
figure,imshow([block_in, block_out]);

%% C2.3
im = im2double(imread('lenna_gray.jpg'));
% Haar transform matrix
N = 16;
haar_4 = make_haar(N);
fcnHandleBlockSplitter = @block_splitter_image;
haar_im = filter_image_haar(im, N, haar_4, fcnHandleBlockSplitter);
figure, imshow(haar_im);

%% C2.4
im = im2double(imread('lenna_gray.jpg'));
% Haar transform matrix
N = 8;
haar_4 = make_haar(N);
fcnHandleBlockSplitter = @block_splitter_image;
haar_im = filter_image_haar(im, N, haar_4, fcnHandleBlockSplitter);
% Reorder the image blocks
im_reorder = reorder_blocks(haar_im, N);
figure,imshow(im_reorder);

%% C2.5
% Lenna Image
im_lena = im2double(imread('lenna_gray.jpg'));
% Haar transform matrix
N = 4;
haar_4 = make_haar(N);
fcnHandleBlockSplitter = @block_splitter_image;
haar_im_lena = filter_image_haar(im_lena, N, haar_4, fcnHandleBlockSplitter);
% Calculate statistics
[im_mean_lena, im_var_lena] = calculate_statistics(haar_im_lena, N);


% Mandrill Image
im_mandril = im2double(imread('mandrill_gray.png'));
% Haar transform matrix
N = 4;
haar_4 = make_haar(N);
fcnHandleBlockSplitter = @block_splitter_image;
haar_im_mandril = filter_image_haar(im_mandril, N, haar_4, fcnHandleBlockSplitter);
% Calculate statistics
[im_mean_mandril, im_var_mandril] = calculate_statistics(haar_im_mandril, N);

% Variances
figure,
plot(log(im_var_lena(:)), 'r'); grid on; hold on;
plot(log(im_var_mandril(:)), 'b');
legend('lena transformed', 'mandrill transformed');

%% C2.6
% Implement blur as a seperable block transform
N = 16;
blur_A = return_blur_A_matrix(N);

% Apply blur matrix to the image
im = im2double(imread('lenna_gray.jpg'));
block_processor_linear = @(block_in, A) A * block_in * A';
blur_lenna = block_splitter_image(im, N, block_processor_linear, blur_A);
figure, imshow([im, blur_lenna]);

%% C2.1
function image_out = block_splitter_image(image_in, block_size, block_func, block_arg)
% Input arguments:
% image_in - input image
% block_size - scalar size of the 2D block (then the resulting block is of dimensions block_sizexblock_size)
% block_func - function handle to apply on each block, takes as input a block and block_arg and returns processed block of the same size
% block_arg - argument provided to block_func (to support notation AXA^t)
% Output arguments:
% image_out - processed image (put together from processed blocks)

% 1. Pad the input image based on the block size
pad_column = block_size - mod(size(image_in, 2), block_size);
pad_row = block_size - mod(size(image_in, 1), block_size);
if pad_column == block_size
    pad_column = 0;
end
if pad_row == block_size
    pad_row = 0;
end
image_in_padded = padarray(image_in, [pad_row, pad_column], 'replicate', 'post');

% 2. Allocate output image variable
image_out = zeros(size(image_in_padded));

% 3. Loop over blocks over x,y with block_size and apply block_func on each block
for x = 1:block_size:(size(image_in_padded, 2) - block_size +1)
    for y = 1:block_size:(size(image_in_padded, 1) - block_size +1)
        block_element = image_in_padded(y:y+block_size-1, x:x+block_size-1);

        % 4. Save processed block onto output image into appropriate part
        image_out(y:y+block_size-1, x:x+block_size-1) = block_func(block_element, block_arg);
    end
end
% 5. Cut away padded areas
image_out(end-pad_row+1: end, :) = [];
image_out(:, end-pad_column+1: end) = [];
end
%% C2.2
function block_out = block_processor(block_in)
%Input argument: block_in - input block, type double in range 0..1
%Output argument: block_out - output flipped block with decreased brightness (type double in range 0..1)

block_out = fliplr(block_in) - 0.3;
block_out(block_out<0) = 0;
end
%% C2.3
function haar_im = filter_image_haar(im, block_size, haar_4, fcnHandleBlockSplitter)
% Input arguments: im - input image (of type double in range 0..1)
% N - block_size
% haar_4 - haar matrix of size 16x16
% fcnHandleBlockSplitter - function handle to block splitter (see problem 1.1 for signature)
% Output argument: haar_im - output image, resulting in applying haar transform to each block

% Describe the block processor
% Haar transform on a block:  A * block_in * A' where A is the transform matrix
block_func = @(block, block_arg) block_arg * block * block_arg';

% Apply block splitter and block processor with the given fcnHandleBlockSplitter
haar_im = fcnHandleBlockSplitter(im, block_size, block_func, haar_4);

end
%% C2.4
function im_reorder = reorder_blocks(im, block_size)
sz = size(im); im_reorder = zeros(sz);
rows = 1: block_size :sz(1); cols = 1: block_size :sz(2);
lr = length(rows); lc = length(cols);
for y = 0: block_size -1
    for x = 0: block_size -1
        block = im(rows+y, cols+x); % 采集每个block的特定位置pixel，汇总成一个新block
        % Block - based scaling
        % block = block - min(block(:));
        % block = block / max(block(:));
        im_reorder(y*lr +1:(y+1)*lr , x*lc +1:(x+1)*lc) = block;
    end
end
end

%% C2.5
function [im_mean, im_var] = calculate_statistics(haar_im, block_size)
% Input argument: haar_im - haar transformed input image of type double in range 0..1
% block_size is a size of the block (so that is of dims (block_size x block_size) )
% Output arguments: im_mean - mean of each coefficient over all blocks (of size block_size x block_size)
% im_var - variances of each coefficient over all blocks (of size block_size x block_size)
sz = size(haar_im);
% 1. Do not forget to pad the input image for correct border treatment
pad = block_size - mod(sz, block_size);
pad(pad == block_size) = 0;
haar_im = padarray(haar_im, pad, 'replicate', 'post');
sz = sz + pad ;
image_out = zeros(size(haar_im));
all_blocks = zeros(block_size, block_size, sz(1)*sz(2) / block_size^2); % dim3: how many results need to be calculated
ibl = 0;

% 2. loop over blocks to get the correct part of each block into a new structure
for x_start = 1: block_size :( sz (2) -block_size +1)
    for y_start = 1: block_size :( sz (1) -block_size +1)
        ibl = ibl + 1;
        block = haar_im(y_start : y_start + block_size -1, x_start : x_start + block_size -1) ;
        all_blocks(1: block_size, 1: block_size, ibl) = block;
    end
end

% 3. calculate stats over a new structure: variance and mean
im_mean = mean(all_blocks, 3);
im_var = var(all_blocks, [], 3);
end

%% C2.6
function blur_A = return_blur_A_matrix(n)
%Input argument: n - size of the filter (so that filter kernel is of size n x n)
%Output argument: blur_A - transformation matrix

% TODO: calculate blur_A as a function of n
blur_A = 0.5* diag(ones(1,n), 0) + 0.25*diag(ones(1,n-1), -1) + 0.25*diag(ones(1,n-1), 1);
end