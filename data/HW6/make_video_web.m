function [video, positions, mask] = make_video_web()
% Create video sequence
% output arguments:
% video - frames organized as a 3D array, where first 2 dimensions are x,y
% position in image and 3rd dimension is time
% positions - positions of an object in this new frame
% mask - mask to indicate position the object within this part of
% the frame, 1 where object pixels are parsed, 0 - background is preserved

%url_tum = 'https://www.dropbox.com/s/v6tow837khkkl3w/tum.jpg?dl=1';
%url_hase = 'https://www.dropbox.com/s/aezfzymczyw5iaw/hase.png?dl=1';

im_tum = im2double(rgb2gray(imread('data/tum.jpg')));
im_tum = imresize(im_tum, 0.5);
[im_obj, ~, mask] = imread('data/hase.png');
im_obj = imresize(im_obj, 0.5);
mask = imresize(mask, 0.5);

im_obj = im2double(rgb2gray(im_obj)) + 0.1;
im_obj(im_obj > 1) = 1;
mask = im2double(mask);
mask = mask > 0.5;
sz_tum = size(im_tum);
sz_obj = size(im_obj);

nframes = 16;
video = zeros([sz_tum, nframes]);
positions = zeros(2, nframes);

for i = 1:nframes
    t = i / nframes;
    b = 1-0.3*t;
    x = round(t*sz_tum(2));
    y = -sin(t*3*2*pi);
    y(y<0) = 0;
    y = round(sz_tum(1) - sz_obj(1) - 10 - 50*y);
    im = paste_mask(b*im_tum, [x y], im_obj, [1 1], mask);
    im(im >1) = 1;
    
    video(:,:,i) = im;
    positions(:,i) = [x;y];
   
end

return