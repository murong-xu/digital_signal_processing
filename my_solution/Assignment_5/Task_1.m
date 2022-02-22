%% Task 1: Autoregressive Models
%% C1.1
im = im2double(imread('lenna_gray.jpg'));
[acov, var, cov_h, cov_v] = autocov(im);
nh2 = 5;
figure, surf(-nh2:nh2, -nh2:nh2, acov); title('Autocovariance'); xlabel('X'); ylabel('Y');

%% C1.2
cov_v_dummy = 0.01;
cov_h_dummy = 0.01;
var_dummy = 0.012;

[ar_gen, rh, rv, std_z] = generate_ar_image(cov_h_dummy, cov_v_dummy, var_dummy);
figure,imshow(ar_gen);

%% A1.2
im = im2double(imread('lenna_gray.jpg'));
[acov, var, cov_h, cov_v] = autocov(im);
[ar_gen, rh, rv, std_z] = generate_ar_image(cov_h, cov_v, var);
figure(3),imshow(ar_gen);
[acov_ar, var_ar, cov_h_ar, cov_v_ar] = autocov(ar_gen);
figure(1), surf(-5:5, -5:5, acov); title('Autocovariance of Lennagray'); xlabel('X'); ylabel('Y');
figure(2), surf(-5:5, -5:5, acov_ar); title('Autocovariance of AR process'); xlabel('X'); ylabel('Y');

%% A1.4
im = im2double(imread('bell-south.jpg'));
[acov, var, cov_h, cov_v] = autocov(im);
nh2 = 5;
figure(1), surf(-nh2:nh2, -nh2:nh2, acov);
title('Autocovariance'); xlabel('X'); ylabel('Y');
[ar_gen, rh, rv, std_z] = generate_ar_image(cov_h, cov_v, var);
figure(2),imshow(ar_gen);
%% C1.1
function [acov, var, cov_h, cov_v] = autocov(im)
% Input argument: im - input image of type double [0..1]
% Output arguments: acov - autocovariance matrix
% var - variance
% cov_h - horizontal covariance
% cov_v - vertical covariance

neighborhood = 11;
nh2 = floor(neighborhood/2); % Neighborhood: nh2 elem. forward/backward
sz = size(im);

% 1. subtract mean
im_centered = im - mean(im(:));

% 2. exclude borders
y_noborder = 1+nh2 : sz(1)-nh2;
x_noborder = 1+nh2 : sz(2)-nh2;

% 3. compute autocovariance matrix
acov = zeros(neighborhood);
for l = -nh2:nh2
    for k = -nh2:nh2
        im_mult = im_centered(y_noborder, x_noborder) .* im_centered(y_noborder+l, x_noborder+k);
        acov(l+nh2+1, k+nh2+1) = mean(im_mult(:));
    end
end

% 4. get variance, covariance in horizontal (one right) and vertical (one below) directions
median_element = ceil(neighborhood/2);
var = acov(median_element, median_element); % self covariance, k=l=0
cov_h = acov(median_element, median_element+1); % k=1, l=0
cov_v = acov(median_element+1, median_element); % k=0, l=1

end

%% C1.2
function [ar_gen, rh, rv, std_z] = generate_ar_image(cov_h, cov_v, var)
% Input arguments: cov_h - covariance in horizontal direction
% cov_v - covariance in vertical direction
% var - variance
% Ouput arguments: ar_gen - generated AR image
% rh - computed rh according to equation 1
% rv - computed rv according to equation 1
% std_z - computed std_z according to equation 1

n = 512; % Size of the image
b = 100; % "Startup" border

% compute parameters rh,rv,std_z
rh = cov_h./var; % k=1,l=0, here rv=1
rv = cov_v./var; % k=0,l=1, here rh=1
std_z = sqrt(var * (1-rh^2) * (1-rv^2));

% allocate and append border
ar_gen = zeros(n+b, n+b);
sz = size(ar_gen);

% generate AR process
z = std_z .* randn(sz); % generate zero-mean Gaussian noise source with variance std_z
% calculate the border pixels specifically
ar_gen(1,1) = z(1,1);
for x_border = 2:sz(2)
    ar_gen(1,x_border) = rh * ar_gen(1,x_border-1) + z(1,x_border);
end
for y_border = 2:sz(1)
    ar_gen(y_border,1) = rv * ar_gen(y_border-1,1) + z(y_border,1);
end
for y = 2:sz(1)
    for x = 2:sz(2)
        ar_gen(y,x) = rh * ar_gen(y,x-1) + rv * ar_gen(y-1,x) - rh * rv * ar_gen(y-1,x-1) + z(y,x);
    end
end

% remove border
ar_gen(1:b, :) = [];
ar_gen(:, 1:b) = [];

end