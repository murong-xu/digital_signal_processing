%% Image manipulation
%% A1.1
im = imread('pears.jpg');
im_type = class(im);
im_dimension = size(im);
figure (1); imhist(im);
im_mean = mean(im(:));
im_max = max(im(:));
im_min = min(im(:));
im_stddev = std(double(im(:)));

%% A1.2
imd = double(im) ./ 255; % double, [0,1]
imd_large = double (im); % double, [0,255]

figure (2); imshow (imd); % works, for type double: [0,1]
figure (3); imshow (im); % works, for type uint8: [0,255]
figure (4); imshow ( imd_large ); % fails

figure (5); colormap (gray (256)); imagesc(im);
figure (6); colormap (gray (256)); imagesc(imd_large);
figure (7); colormap (gray (256)); imagesc(imd); % imagesc() is linear mapping
figure (8); colormap (gray (256)); image(imd); % fails, image() is absolut mapping, imd in range of [0,1], not match

figure (9); colormap (gray (256)); % Image as 3D surface
surface ([0 1],[0 1],0.5 * ones (2),im,'FaceColor','texturemap','CDataMapping','direct');
imtool (imd);

%% C1.1
im_random = abs(randn(64,64));
imd_frame = im_add_frame(im_random);
figure,imshow(imd_frame); % imshow() does not change the size

function imd_frame = im_add_frame (im)
frame_color = 0;
im([1:2, end - 1: end], :) = frame_color;
im(:, [1:2, end - 1: end]) = frame_color;
imd_frame = im;
end

%% C1.2
im = abs(randn(512,512));
im_part = im_cut_part(im);
figure,imshow(im_part);

function im_part = im_cut_part(im)
% Input argument: im - grayscale image
% Output argument: im_part - output cut image
im = im([1:2:end],[1:3:end]);
im_part = im;
end

%% A1.4
imd = double(imread('pears.jpg')) ./ 255;
im_dimension = size(imd);
figure; imshow(imd);
hold on; % Allows to add more elements to figure
x = 1:5:200;
y = 400 .* ones(length(x),1);
handle = plot(x, y, '--r', 'LineWidth', 20);
pause(3); % Change plot object after 3 seconds :
set(handle, 'Color', 'green', 'XData', x+100, 'YData', 0.8*y);

%% C1.3
imd = double(imread('pears.jpg')) ./ 255;
im_kron_rgb = merge_channels_kronecker_4(imd);
figure,imshow(im_kron_rgb);

% method 1: using cell matrix
function im_kron_rgb = merge_channels_kronecker_1(im)
cR = [1, 0, 1; 0.3, 1, 0; 0, 0.3, 1];
cG = [0, 1, 1; 0.3, 0, 1; 0.7, 0.3, 0];
cB = [0, 0, 0; 1, 1, 0; 0.7, 1, 0];
[M,N] = size(cR);
[U,V] = size(im);
im_kron_r = cell(M, N);
im_kron_g = cell(M, N);
im_kron_b = cell(M, N);
for kk = 1: M
    for ii = 1: N
        im_kron_r{kk,ii} = im .* cR(kk,ii);
        im_kron_g{kk,ii} = im .* cG(kk,ii);
        im_kron_b{kk,ii} = im .* cB(kk,ii);
    end
end
im_kron_rgb = zeros(M.*U, N.*V, 3);
im_kron_rgb(:,:,1) = cell2mat(im_kron_r);
im_kron_rgb(:,:,2) = cell2mat(im_kron_g);
im_kron_rgb(:,:,3) = cell2mat(im_kron_b);
end

% method 2: using sub-matrix addressing
function im_kron_rgb = merge_channels_kronecker_2(im)
cR = [ 1 0 1 ; 0.3 1 0; 0 0.3 1 ];
cG = [ 0 1 1 ; 0.3 0 1; 0.7 0.3 0 ];
cB = [ 0 0 0 ; 1 1 0; 0.7 1 0 ];
[M,N] = size (im);
im_kron_r = zeros (3* [M,N]);
im_kron_g = zeros (3* [M,N]);
im_kron_b = zeros (3* [M,N]);
row = 1:M;
for row_counter = 1:3
    column = 1:N;
    for column_counter = 1:3
        im_kron_r (row, column) = cR(row_counter,column_counter) .* im;
        im_kron_g (row, column) = cG(row_counter,column_counter) .* im;
        im_kron_b (row, column) = cB(row_counter,column_counter) .* im;
        column = column + N;
    end
    row = row + M;
end
im_kron_rgb = zeros (3 * M, 3 * N, 3);
im_kron_rgb (: ,: ,1) = im_kron_r ;
im_kron_rgb (: ,: ,2) = im_kron_g ;
im_kron_rgb (: ,: ,3) = im_kron_b ;
end

% method 3: using sub-matrix addressing, tutor version
function im_kron_rgb = merge_channels_kronecker_3(im)
cR = [ 1 0 1 ; 0.3 1 0; 0 0.3 1 ];
cG = [ 0 1 1 ; 0.3 0 1; 0.7 0.3 0 ];
cB = [ 0 0 0 ; 1 1 0; 0.7 1 0 ];
sz = size (im);
im_k_R = zeros (3* sz); im_k_G = zeros (3* sz); im_k_B = zeros (3* sz);
for r = 1:3
    for c = 1:3
        range_r = (r -1) *sz (1) +1 : r*sz (1) ;
        range_c = (c -1) *sz (2) +1 : c*sz (2) ;
        im_k_R ( range_r , range_c ) = cR(r,c) * im;
        im_k_G ( range_r , range_c ) = cG(r,c) * im;
        im_k_B ( range_r , range_c ) = cB(r,c) * im;
    end
end
im_kron_rgb = zeros ([3* sz 3]);
im_kron_rgb (: ,: ,1) = im_k_R ;
im_kron_rgb (: ,: ,2) = im_k_G ;
im_kron_rgb (: ,: ,3) = im_k_B ;
end

% method 4: using built-in function kron()
function im_kron_rgb = merge_channels_kronecker_4(im)
cR = [ 1 0 1 ; 0.3 1 0; 0 0.3 1 ];
cG = [ 0 1 1 ; 0.3 0 1; 0.7 0.3 0 ];
cB = [ 0 0 0 ; 1 1 0; 0.7 1 0 ];
sz = size (im);
im_k_R = kron(cR, im);
im_k_G = kron(cG, im);
im_k_B = kron(cB, im);
im_kron_rgb = zeros ([3* sz 3]);
im_kron_rgb (: ,: ,1) = im_k_R ;
im_kron_rgb (: ,: ,2) = im_k_G ;
im_kron_rgb (: ,: ,3) = im_k_B ;
end