%% Task 2: Karhunen-Loeve-Transform
%% C2.1
im = im2double(imread('lenna_gray.jpg'));
block_splitter = @block_splitter_image;
im_reshape = reshape_image(im, block_splitter);
figure,imshow(im_reshape);

%% C2.2
im = im2double(imread('lenna_gray.jpg'));
[blocks, C_im, klt_base] = compute_klt_basis(im);
figure,imshow([klt_base, C_im]);

%% A2.1
im1 = im2double(imread('lenna_gray.jpg'));
[blocks_1, C_im_1, klt_base_1] = compute_klt_basis(im1);
im2 = im2double(imread('pears.jpg'));
[blocks_2, C_im_2, klt_base_2] = compute_klt_basis(im2);
im3 = im2double(imread('mandrill_gray.png'));
[blocks_3, C_im_3, klt_base_3] = compute_klt_basis(im3);
figure(1), imshow(klt_base_1);
figure(2), imshow(klt_base_2);
figure(3), imshow(klt_base_3);

%% C2.3
im = im2double(imread('lenna_gray.jpg'));
block_splitter = @block_splitter_image;
[blocks, C_im, klt_base_v] = compute_klt_basis(im);  
[im_projected, im_backprojected, klt_psnr] = project_reconstruct_klt_basis(im, klt_base_v, block_splitter);
figure,imshow([im, im_backprojected]);
fprintf('PSNR %f\n', klt_psnr);

%% C2.4
% 1. Calculate the autocorrelation of KLT transformed Lenna image (KLT basis images are trained from Lenna image)
im_lenna = im2double(imread('lenna_gray.jpg'));
% KLT basis images of Lenna image
load('klt_base_v.mat', 'klt_base_v');
block_splitter = @block_splitter_image;
[im_projected_lenna, im_backprojected_lenna, klt_psnr_lenna] = project_reconstruct_klt_basis(im_lenna, klt_base_v, block_splitter);
N = 8;
autocorr_lenna = compute_autocorrelation_klt_images(im_projected_lenna, N);
figure,imagesc(log(abs(autocorr_lenna))); colormap(jet);


% 2. Calculate the autocorrelation of KLT transformed Mandrill image (KLT basis images are trained from Lenna image)
% Here we project the Mandrill image onto the KLT basis images trained from Lenna image
im_mandrill = im2double(imread('mandrill_gray.png'));
% KLT basis images of Lenna image
load('klt_base_v.mat', 'klt_base_v');
block_splitter = @block_splitter_image;
[im_projected_mandrill, im_backprojected_mandrill, klt_psnr_mandrill] = project_reconstruct_klt_basis(im_mandrill, klt_base_v, block_splitter);
N = 8;
autocorr_mandrill = compute_autocorrelation_klt_images(im_projected_mandrill, N);
figure,imagesc(log(abs(autocorr_mandrill))); colormap(jet);

%% C2.1
function im_reshape = reshape_image(im, block_splitter)
% Input argument: im - input image
% block_splitter - function handle of the block splitter
% Ouput argument: im_reshape - reshaped image
N = 32;%size of the block
v = zeros(N^2,1);
%implement reshaping here
v([529:544, 272:32:784]) = 0.1;
func_handle = @(block, argv)reshape(block(:)+argv, [N,N]); % just preserve the block, change this!
im_reshape = block_splitter(im, N, func_handle, v);
end

%% C2.2
function [blocks, C_im, klt_base] = compute_klt_basis(im)
% Input arguments: im - input image
% Output arguments: blocks - 2D structure containing blocks of image
% C_im - autocorrelation matrix of an image for each coefficient over all blocks
% klt_base - basis images for this KLT transformation

N = 8; % N - size of the block

% 1. Allocate memory for blocks
sz = size(im);
x_arr = 1:N:(sz(2) - N+1);
y_arr = 1:N:(sz(1) - N+1);
blocks = zeros(length(x_arr) * length(y_arr), N*N);

%2. Regroup the input image into blocks
% do not forget, y goes along the first dimension (rows) of the image matrix, x goes along columns
% also important: we use row-major ordering!
for iy = 1:length(y_arr)
    for ix = 1:length(x_arr)
        % compute the blocks here
        block_temp = im(y_arr(iy):y_arr(iy)+N-1, x_arr(ix):x_arr(ix)+N-1);
        blocks((iy-1)*length(x_arr)+ix, :) = block_temp(:);
    end
end

%3. Compute autocorrelation over all blocks
C_im = (blocks' * blocks) ./ size(blocks, 1);

%4. Compute eigenvalue decomposition of the autocorrelation matrix and thus get the KLT basis
[V,D] = eig(C_im); % V: cols are eigenvectors, D: diag. matrix with eigenvalues
[D_sort,index] = sort(diag(D),'descend');
D_sort = D_sort(index);
V_sort = V(:,index);

% pack basis vectors to 64x64 basis image matrix (col major)
klt_base = zeros(N^2, N^2);
for index_y = 0:N-1
    for index_x = 0:N-1
        klt_base(index_y*N+1:(index_y+1)*N, index_x*N+1:(index_x+1)*N) = reshape(V_sort(:,index_x*N + index_y+1), [N,N]);
    end
end

end

%% C2.3
function [im_projected, im_backprojected, klt_psnr] = project_reconstruct_klt_basis(im, klt_base_v, block_splitter)
%% Input arguments:
% im - input image
% klt_base_v - KLT basis (as obtained before)
% block_splitter - block splitter function handle (signature as used before in HW3)
%% Output arguments:
% im_projected - image projected onto KLT basis
% im_backprojected - back-projected image (e.g. projected onto original basis after projecting onto KLT basis)
% klt_psnr - PSNR of the difference image (e.g. of the image im_backprojected - im)

% 1.a First define block processor for orthogonal base projection
    function block_out = block_processor_orth_base_proj(block_in, base) % block_in: NxN, block_in(:): N.^2x1
        block_size = size(block_in);
        block_out_temp = base' * block_in(:);
        block_out = reshape(block_out_temp, block_size); % TODO implement here
    end

% 1.b. Define block processor for othogonal base backprojecting
    function block_out = block_processor_orth_base_backproj(block_in, base)
        block_size = size(block_in);
        block_out_temp = base * block_in(:);
        block_out = reshape(block_out_temp, block_size); % TODO implement here
    end

% 2. Then using a block splitter project the image
base_image_size = size(klt_base_v, 1);
N = sqrt(base_image_size);
im_projected = block_splitter(im, N, @block_processor_orth_base_proj, klt_base_v);

% 3. Then backproject to the original image
im_backprojected = block_splitter(im_projected, N, @block_processor_orth_base_backproj, klt_base_v);

% 4. And compute PSNR of the difference
im_diff = im_backprojected - im;
klt_psnr = 10 * log10(1/ mean(im_diff(:).^2));
end

%% C2.4
function autocorr = compute_autocorrelation_klt_images(klt_projected_image, N)
% Input arguments: klt_projected_image - KLT blocks
% N - size of the block
% Output arguments: autocorr - autocorrelation matrix

%1. First collect blocks of transformed images
sz = size(klt_projected_image);
x_arr = 1:N:(sz(2) - N+1);
y_arr = 1:N:(sz(1) - N+1);
blocks = zeros(length(x_arr) * length(y_arr), N*N);

for iy = 1:length(y_arr)
    for ix = 1:length(x_arr)
        % compute the blocks here
        block_temp = klt_projected_image(y_arr(iy):y_arr(iy)+N-1, x_arr(ix):x_arr(ix)+N-1);
        blocks((iy-1)*length(x_arr)+ix, :) = block_temp(:);
    end
end
%2. Then calculate autocorrelation matrix
autocorr = (blocks' * blocks) ./ size(blocks, 1);
end

%% tool function splitter
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