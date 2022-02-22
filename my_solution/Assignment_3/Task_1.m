%% Task 1: Discrete Fourier Trnasform
%% C1.1
im = im2double(imread('lenna_gray.jpg'));
sig1 = 5;
sig2 = 9;
N = 16; % Size of the filter, not odd number as usual...
[im_freq_filt_spat_2, im_freq_filt_freq] = filter_frequency_domain(im, N, sig1,sig2);
figure, imshow(abs(im_freq_filt_spat_2)*10);

%% A1.2
im = im2double(imread('lenna_gray.jpg'));
im_saltpepper = imnoise(im, 'salt & pepper', 0.05);
im_filtered = medfilt2(im_saltpepper);  % 2D medien filtering
figure(1), imshow(abs(im));
figure(2), imshow(abs(im_saltpepper));
figure(3), imshow(abs(im_filtered));

%% C1.2
im = im2double(imread('lenna_gray.jpg'));
imf = fft2(im);
conj_point_symm1 = is_conj_point_symm(imf);
assert(conj_point_symm1); % if no message is shown, then you are fine

%% C1.3
im = im2double(imread('lenna_gray.jpg'));
x0 = 50;
x_c = x0:120;
y0 = 70;
y_c = y0:140;
ims = im(y_c,x_c);
[x,y] = find_part_location_image(im, ims);
figure,imshow(im); title(['Found location ', num2str(x), 'x', num2str(y)]);
figure,imshow(ims); title(['Correct location ', num2str(x0), 'x', num2str(y0)]);

%% C1.1
function [im_freq_filt_spat, im_freq_filt_freq] = filter_frequency_domain(im, N, sig1, sig2)
% Input arguments im - image
% N size of the Gaussian filter kernel
% sig1 - standard deviation of the first gaussian filter
% sig2 - standard deviation of the second gaussian filter
% Output arguments:
% im_freq_filt_spat - spatial representation of filtering the image in frequency domain
% im_freq_filt_freq - frequency representation of filtering the image in frequency domain

% Calculate G in frequency domain
g1 = fspecial('gaussian', N, sig1);
g2 = fspecial('gaussian', N, sig2);
g = g1 - g2;

% As the filter kernel is smaller than the Image,
% need to apply padding with zeros on the filter kernel
g_pad = padarray(g, ([size(im,1) size(im,2)] - size(g)), 'post');

% Then, another important step is to apply circularshift on the g_pad
% such that the center of the filter kernel is located at index(1,1)
g_pad = circshift(g_pad, - floor(size(g)/2));
G = fft2(g_pad);

% Filter in frequency domain
% 1. Take the fft2 of the image and return as imf
imf = fft2(im);
% 2. Multiply the imf with G and return as the im_freq_filt_freq
im_freq_filt_freq = G .* imf;  % convolution in spatial domain = multiplication in frequency domain
% 3. Take the ifft2 of im_freq_filt_freq and return as the im_freq_filt_spat
im_freq_filt_spat = ifft2(im_freq_filt_freq);
end

%% C1.2
function conj_point_symm = is_conj_point_symm(imf)
% Input argument: imf - frequency transform of an image
% Output argument: conj_point_symm - true if conjugate point-symmetry holds for this image,
% and false otherwise

% Use fftshift to have the zero frequency component in the center of the spectrum
imf_shifted = fftshift(imf);

% Copy the first row/col for to the end+1^th row/col
imf_shifted(:, end+1) = imf_shifted(:,1);
imf_shifted(end+1, :) = imf_shifted(1,:);

% Get upper & lower matrix, check point-symmetry:
imf_upper = imf_shifted(1:round(size(imf_shifted, 1)/2), 1:end);
imf_lower = imf_shifted(round(size(imf_shifted, 1)/2):end, 1:end);
% x = fliplr(imf_lower);
% x = flipud(x);
x = rot90(imf_lower, 2);

diff = imf_upper - conj(x);
% if any(diff ~= 0)
%     conj_point_symm = 0
% else
%     conj_point_symm = 1
% end
is_conj_symm = abs(diff) < 1e-14;
conj_point_symm = all(is_conj_symm(:));
end
%% C1.3
function [x,y] = find_part_location_image(im, ims)
% Input argument:
% im - input image of original size
% ims - part of the original image of smaller size (i.e. cut out part of im)
% Output arguments: x and y - shift coordinates

% Pad the ims with zeros to have the same size with the original image
ims_padded = padarray(ims, size(im)-size(ims), 'post');

% Use the given formula to calculate the spatial shift
I1 = fft2(ims_padded);
I2 = fft2(im);
cross_Power_Spectrum = I1 .* conj(I2) ./ abs(I1 .* conj(I2));
t = ifft2(conj(cross_Power_Spectrum));
% max_element = max(t(:));
% max_ind = find(t==max_element);
[max_element, max_ind] = max(t(:));
[y_temp,x_temp] = ind2sub(size(im), max_ind);
x = x_temp;
y = y_temp;
end