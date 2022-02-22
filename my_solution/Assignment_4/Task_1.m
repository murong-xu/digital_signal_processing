%% Task 1: Discrete Cosine Trnasform
%% C1.1
image_in = im2double(imread('lenna_gray.jpg'));
% Notation for block_splitter_image(image_in, block_size, block_func, block_arg)
fcnHandleBlkSplitter = @block_splitter_image;
M = 16;
dct_matrix = dctmtx(M);
dct_image = transform_to_dct_domain(image_in, dct_matrix, fcnHandleBlkSplitter);
% Scaling
% dct_image = dct_image - min(dct_image(:));
% dct_image = dct_image / max(dct_image(:));
figure,imshow(dct_image);

%% A1.1
image_in = im2double(imread('lenna_gray.jpg'));
% Notation for block_splitter_image(image_in, block_size, block_func, block_arg)
fcnHandleBlkSplitter = @block_splitter_image;
M = 16;
dct_matrix = dctmtx(M);
dct_image = transform_to_dct_domain(image_in, dct_matrix, fcnHandleBlkSplitter);

[im_mean, im_var] = calculate_statistics(image_in, M);
[dct_mean, dct_var] = calculate_statistics(dct_image, M);
figure(1);
surf(log(im_var));
view(60, 30);
zlabel('log(Variance) of original image');
xlabel('Block element x');
ylabel('Block element y');
figure(2);
surf(log(dct_var));
view(60, 30);
zlabel('log(Variance) after DCT transformation');
xlabel('Block element x');
ylabel('Block element y');

%% C1.2
image_in = im2double(imread('lenna_gray.jpg'));
fcnHandleBlkSplitter = @block_splitter_image;
M = 16;
dct_matrix = dctmtx(M);
% dct transformed image
dct_image = transform_to_dct_domain(image_in, dct_matrix, fcnHandleBlkSplitter);
% Compress the image
N = 16; %number of DCT coefficients that will be kept
im_rec_comp = compress_dct(dct_image, dct_matrix, fcnHandleBlkSplitter, N);
figure,imshow([image_in, im_rec_comp]);

%% C1.3
im_in = zeros(2,2,2);
im_in(:,:,1) = [2 1; 2 0];
im_in(:,:,2) = [3 1; 2 0];
dct2 = dctmtx(2);
im_dct = apply_3ddct(im_in, dct2)

%% C1.4
fcnHandle3DDCTTransformation = @apply_3ddct;
dct_mat = dctmtx(2);
load('video_public_mat_short.mat');
[video_dc, video_hf] = threed_dct_analysis(video_public_mat_short, dct_mat, fcnHandle3DDCTTransformation); % TODO visualize this sequence as you wish

%% C1.1
function dct_image = transform_to_dct_domain(im, dct_matrix, fcnHandleBlkSplitter)
% Input arguments: im - input image of type double in range 0..1
% dct_matrix - DCT transform matrix
% fcnHandleBlkSplitter - function handle to block splitter with arguments (im, block size,
% function handle to block transform, function arguments to block transform)
% Output argument: dct_image - dct transformed image

linear_block_transform = @(block_in, A)A * block_in * A'; % TODO change this to the correct block transform
block_sz = size(dct_matrix,1);
dct_image = fcnHandleBlkSplitter(im, block_sz, linear_block_transform, dct_matrix);
end

%% C1.2
function im_rec_comp = compress_dct(dct_image, dct_matrix, fcnHandleBlkSplitter, N)
% input arguments: dct_image - image transformed to dct domain
% dct_matrix - DCT transformation matrix for block size 16
% fcnHandleBlkSplitter - function handle to split matrix into blocks, its signature is same as in Assignment 3, i.e.
% image_out = block_splitter(image_in, block_size, block_func, varargin)
% N - number of DCT coefficients that will be kept
% im_rec_comp reconstructed image using only N first DCT coefficients

block_size = 16;
block_processor_linear = @(block_in, A) A * block_in * A';

% 1. remove all but first N coefficients
A_new = zeros(block_size, block_size);
for i = 1:sqrt(N)
    A_new(i,i) = 1;
end
dct_new = fcnHandleBlkSplitter(dct_image, block_size, block_processor_linear, A_new);

% 2. back-transform the image using given function handles
im_rec_comp = fcnHandleBlkSplitter(dct_new, block_size, block_processor_linear, dct_matrix'); % dct matrix is real-valued
end

%% C1.3
function im_dct = apply_3ddct(im_in, dct_mat)
% input arguments: in_in - input 3D signal (2x2x2)
% dct_mat - 2D DCT transformation matrix (2x2)
% output arguments: im_dct - transformed DCT 3D signal (2x2x2)

% 1. vertical columns
im_vertical = dct_mat * reshape(im_in, [2,4]);
im_vertical = im_vertical';
im_vertical_out = [im_vertical(1:2, :), im_vertical(3:4, :)];
% 2. horizontal rows
im_horizontal = dct_mat * im_vertical_out;
im_horizontal = reshape(im_horizontal, [1,8]);
im_horizontal_out = [im_horizontal(:, 5:8); im_horizontal(:, 1:4)];
% 3. out-of-plane rotations
im_plane = dct_mat * im_horizontal_out;

% back transpose
im_trans_hor = [reshape(im_plane(2,:), [2,2]), reshape(im_plane(1,:), [2,2])];
im_dct = [im_trans_hor(:, 1:2)', im_trans_hor(:, 3:4)'];
im_dct = reshape(im_dct, [2,2,2]); % don't forget this is a 2x2x2 matrix
end

%% C1.4
function [video_dc, video_hf] = threed_dct_analysis(video, dct_mat, fcnHandle3DDCTTransformation)
% input arguments: video - 3D sequence (3rd dimension corresponds to time) of size sz_i
% dct_mat - 2D DCT transformation matrix
% fcnHandle3DDCTTransformation - function handle to 3D dct transformation with
% arguments (im_in, dct_mat)
% output arguments: video_dc - DC component of DCT transformed sequence (dimensions sz_i./blockSize)
% video_hf - high frequency component of DCT transformed sequence (dimensions sz_i./blockSize)
blockSize = 2;
sz_v = size(video);
video_out = zeros(sz_v); % result of DCT-transformation
video_dc = zeros(sz_v./blockSize);
video_hf = zeros(sz_v./blockSize);

% Here you need to iterate along first, second and third dimensions of the video sequence and
% apply DCT in a block-wise fashion
step = blockSize - 1;
for col=1: blockSize : sz_v(1)-1
    for row=1: blockSize : sz_v(2)-1
        for plane=1: blockSize : sz_v(3)-1
            temp = video(col:col+step,row:row+step,plane:plane+step);
            video_out(col:col+step,row:row+step,plane:plane+step) = fcnHandle3DDCTTransformation(temp, dct_mat);
        end
    end
end

video_dc = video_out(1:blockSize:end, 1:blockSize:end, 2:blockSize:end);
video_hf = video_out(2:blockSize:end, 2:blockSize:end, 1:blockSize:end);

end

%% function spiltter
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

%% function calculate_statistics
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