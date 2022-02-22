%% Task3: Image Filters
%% A3.1
Hg = fspecial('average',3);
Hg1 = fspecial('gaussian', 3, 0.5);
figure ; freqz2( Hg1); title('gaussian -3 -0 .5 ');
Hg2 = fspecial('gaussian', 8, 0.5);
figure ; freqz2( Hg2); title('gaussian -8 -0 .5 ');
Hg3 = fspecial('gaussian', 8, 3);
figure ; freqz2( Hg3); title('gaussian -8 -3 ');
Hg4 = fspecial('gaussian', 16, 3);
figure ; freqz2( Hg4); title('gaussian -16 -3 ');
Hg5 = fspecial('gaussian', 16, 5);
figure ; freqz2( Hg5); title('gaussian -16 -5 ');
Hg6 = fspecial('gaussian', 8, 1.5);
figure ; freqz2( Hg6); title('gaussian -8 -1 .5 ');
Hlog1 = fspecial('log');
figure ; freqz2( Hlog1 ); title('log');
Hp1 = fspecial('prewitt');
figure ; freqz2( Hp1); title('prewitt');
Hl1 = fspecial('laplacian');
figure ; freqz2( Hl1); title('laplacian');
Hd1 = fspecial('disk' ,3);
figure ; freqz2( Hd1); title('disk');
figure ; imshow( Hd1/ max( Hd1(:)), 'InitialMagnification', 2000) ;
title('disk');

%% C3.1
z_values = prewitt_z();
X = (-32:31) / 32;
[Xm,Ym] = meshgrid(X);
figure,mesh(Xm, Ym, z_values);title('prewitt with fft2 (your result)');
xlabel('X'); ylabel('Y');

%% C3.2
im = im2double(imread('pears.jpg'));
im_filtered = filter_image(im);
figure,imshow(im);
figure,imshow(im_filtered);

%% A3.2
Hg1 = fspecial('gaussian', 3, 0.5);
Hg2 = fspecial('gaussian', 3, 5);
Hg3 = fspecial('gaussian', 16, 0.5);
Hg4 = fspecial('gaussian', 16, 5);
Hg5 = fspecial('sobel');

im = im2double(imread('pears.jpg'));
im_filtered_1 = imfilter(im, Hg1);
im_filtered_2 = imfilter(im, Hg2);
im_filtered_3 = imfilter(im, Hg3);
im_filtered_4 = imfilter(im, Hg4);
im_filtered_5 = imfilter(im, Hg5);
figure(1),imshow(im); title('original');
figure(2),imshow(im_filtered_1); title('gaussian size=3  sigma=0.5');
figure(3),imshow(im_filtered_2); title('gaussian size=3  sigma=5');
figure(4),imshow(im_filtered_3); title('gaussian size=16  sigma=0.5');
figure(5),imshow(im_filtered_4); title('gaussian size=16  sigma=5');
figure(6),imshow(im_filtered_5); title('sobel');

%% C3.3
N = 350;
[D, k, Izp] = calculate_zone_plate(N);
[diag_filter_subsampled, diag_nonfilter_subsampled] = filter_subsample_zone_plate(Izp);
figure, plot(diag_filter_subsampled, 'r'), hold on; plot(diag_nonfilter_subsampled, 'b'); legend('Filtered', 'Not filtered');

%% A3.3

%% C3.1
function prew_z = prewitt_z()
% Output argument:
% prew_z - absolute value of a complex number used for a frequency plot within freqz2()
N=64;% size of the filter is 64
% implement how to calculate prew_z using fft2() function
h = fspecial('prewitt');
H = fft2(h, N, N);
prew_z = abs(fftshift(H));
end

%% C3.2
function img_filtered = filter_image(img)
% Input argument: img - grayscale image, of type double in range 0..1
% Output argument: img_filtered - filtered grayscale image, of type double in range 0..1
% implement image filtering
hsize = 16;
sigma = 3;
H = fspecial('gaussian', hsize, sigma);
img_filtered = imfilter(img, H, 'replicate');
end

%% C3.3
function [diag_filter_subsampled, diag_nonfilter_subsampled] = filter_subsample_zone_plate(Izp)
% Input arguments:
% Izp-grayscale image of the zone plate of type double in range 0..1
% Output arguments:
% diag_filter_subsampled - main diagonal of the filtered and subsampled image
% diag_nonfilter_subsampled-main diagonal of the non-filtered and subsampled image

% for first output filter and sub-sample Izp
H = fspecial('gaussian', 16, 3);
Izp_filtered = imfilter(Izp, H, 'symmetric');
Izp_filtered = Izp_filtered(1:2:end, 1:2:end);
diag_filter_subsampled = diag(Izp_filtered);

% for second output just sub-sample Izp
Izp_sampled = Izp(1:2:end, 1:2:end);
diag_nonfilter_subsampled = diag(Izp_sampled);
end

function [D, k, Izp] = calculate_zone_plate(N)
% input argument:
% N - side length of the image matrix D
% output arguments:
% D is the distance matrix from element (1,1)
% k - computed value of k
% Izp image of the zone plate

% add your code to calculate image matrix D
[X, Y] = meshgrid(1:N);
X = X - 1;  X = X.^2;
Y = Y - 1;  Y = Y.^2;
D = sqrt(X + Y);

% and to compute k
k = 1 / (D(end, end)^2 - D(end-1, end-1)^2);

% and to compute Izp - zone plate image
Izp_intermediate = 2 * pi * k./2 * D.^2;
Izp = 0.5 + 0.5 * cos(Izp_intermediate);
end
%% A3.3