%% Periodic Sequences
%% C2.1
pm1 = [200 0; 0 200];
pm2 = [200 200; 0 200];
pm3 = [200 200; -200 200];
loc_base = [1, 1];
im_ref = im2double(imread('texture1.jpg'));
[atomic, mask] = get_atomic(im_ref, pm3, loc_base);
figure(1),imshow(mask);
figure(2),imshow(atomic);
%% C2.2
im_atomic = atomic;
atomic_mask = mask;
size_output = [600, 800];
loc_base_2 = [1;1];
im_out = create_periodic(im_atomic, atomic_mask, pm3, loc_base_2, size_output);
figure(3),imshow(im_out);
%% C2.3
i1 = abs(randn(100,100));
i2 = abs(i1);
ncc = get_ncc(i1,i2);
%% C2.4
fcnHandleNCC = @get_ncc;
range=1:300;
im = im2double(imread('texture2.jpg'));
[ncc_range1, ncc_max_loc1] = compute_ncc_find_max(im, range, fcnHandleNCC);
figure,plot(ncc_range1);
if isempty(ncc_max_loc1)==0
    ncc_max_loc1
end

function [atomic, mask] = get_atomic(im_ref, pm, loc_base)
% function to get atomic element for a given image im_ref
% Input arguments:
% im_ref - reference image of type double in range 0..1
% loc_base - starting location (size 1x2)
% pm - periodic matrix (size 2x2, where each column corresponds to a periodicity vector)
% Output arguments:
% atomic - atomic element of the image  of type double in range 0..1
% mask - corresponding mask for the atomic element

% 1. Define polygon (where first column is [0; 0]; the other columns are
% computed by using columns of pm using addition/subtraction):
poly = zeros(2,5);
% ENTER SOLUTION HERE
poly (: ,1) = [0 0]';
poly (: ,2) = poly (: ,1) + pm (: ,1);
poly (: ,3) = poly (: ,2) + pm (: ,2);
poly (: ,4) = poly (: ,3) - pm (: ,1);
poly (: ,5) = poly (: ,4) - pm (: ,2);

% 2. Creating mask for atomic element (parallelogram) with poly2mask
% here we shift the polygon by some sub-pixel fraction to get the required
% number of pixels, DO NOT EDIT the two lines below
poly = poly + repmat([0; 0.4] - min(poly, [], 2), 1, 5); % bring the most left top point upper 0.4
sz_mask = round(max(poly, [], 2)); % shift the polygon upper 0.4, computing the size of the mask

% compute mask using poly2mask
% ENTER SOLUTION HERE
mask = poly2mask ( poly (1 ,:) , poly (2 ,:) , sz_mask (2) , sz_mask (1));
% Check size / area
if sum ( mask (:)) == det(pm); fprintf ('Area : %upx - ok\n',sum ( mask (:)));
else ; fprintf ('Area : % upx - error \n', sum( mask (:)));
end

% 3. Compute atomic element
% ENTER SOLUTION HERE
atomic = im_ref ( loc_base (2): loc_base (2)+ sz_mask (2) -1, loc_base (1):loc_base (1)+ sz_mask (1) -1);
atomic = atomic .* mask ;
end

%% C2.2
function im_out = create_periodic(im_atomic, atomic_mask, pm_mat, loc_base, size_output)
%% Input arguments:
% im_atomic - atomic element of the image
% atomic_mask - mask corresponding to the atomic element
% pm_mat - periodicity matrix
% loc_base - base location (x,y) (1x2) of the atomic element in the original
% size_output - needed size of the output
%
%% Output arguments
% im_out - output periodic image
im_out = zeros(size_output);

% YOUR SOLUTION HERE
% Calculate required range for nx, ny
loc_atomic_shifted = [1,size_output(2), size_output(2),1; 1,size_output(1),1, size_output(1)] - repmat(loc_base,1,4);
coordinate_tranf_matrix = inv(pm_mat);
vertex_left_top = coordinate_tranf_matrix * loc_atomic_shifted(:,1);
vertex_left_bottom = coordinate_tranf_matrix * loc_atomic_shifted(:,4);
vertex_right_top = coordinate_tranf_matrix * loc_atomic_shifted(:,3);
vertex_right_bottom = coordinate_tranf_matrix * loc_atomic_shifted(:,2);
vertex_x_range = [vertex_left_top(1),vertex_left_bottom(1),vertex_right_top(1),vertex_right_bottom(1)];
vertex_y_range = [vertex_left_top(2),vertex_left_bottom(2),vertex_right_top(2),vertex_right_bottom(2)];
nx_range_min = min(vertex_x_range);
nx_range_max = max(vertex_x_range);
ny_range_min = min(vertex_y_range);
ny_range_max = max(vertex_y_range);
nx_range = floor(nx_range_min) - 1: ceil(nx_range_max) + 1;  % -1/+1 ensures what?
ny_range = floor(ny_range_min) - 1: ceil(ny_range_max) + 1;

% "Copy-Paste" the atomic element multiple times, add to output image
for nx = nx_range
    for ny = ny_range

        % YOUR SOLUTION HERE

        % Calculate pixel positions top-left and bottom-right:
        loc_top = loc_base + nx * pm_mat(:,1) + ny * pm_mat(:,2);
        loc_bottom = loc_top + [size(im_atomic,2); size(im_atomic,1)] - 1;
        loc_x = loc_top(1): loc_bottom(1);
        loc_y = loc_top(2): loc_bottom(2);
        % Remove cols/rows outside matrix:
        x_exist = loc_x;
        x_exist(x_exist <1)=0;
        x_exist(x_exist > size_output(2))=0;
        x_exist(x_exist ~= 0) = 1;
        x_exist = logical(x_exist);

        y_exist = loc_y;
        y_exist(y_exist <1)=0;
        y_exist(y_exist > size_output(1))=0;
        y_exist(y_exist ~= 0) = 1;
        y_exist = logical(y_exist);

        if any(x_exist) || any(y_exist)
            % Copy the atomic element with transparency mask
            index_x = 1: size(im_atomic, 2);
            index_y = 1: size(im_atomic, 1);
            add_new_element = atomic_mask(index_y(y_exist), index_x(x_exist)) .* im_atomic(index_y(y_exist), index_x(x_exist));
            im_out(loc_y(y_exist), loc_x(x_exist)) = add_new_element + im_out(loc_y(y_exist), loc_x(x_exist));
            %            figure(1),imshow(im_out);
            %         hold on
        end
    end
end
end

%% C2.3
function ncc = get_ncc(i1, i2)
% Calculate NCC for equal-sized t1, t2
% implement the task here
sum_variable = (i1 - mean(i1(:))) .* (i2 - mean(i2(:))) ./ (std(i1(:)) * std(i2(:)));
sum_result = sum(sum_variable(:));
ncc = 1/(numel(i1)) * sum_result;
end

%% C2.4
function [ncc_range, ncc_max_loc] = compute_ncc_find_max(im, range, fcnHandleNCC)
% Input arguments:
% im - given image (of type double in range 0..1)
% range - vector containing elements from 1 till 300
% fcnHandleNCC - function handle to function "get_ncc(t1,t2)" -> see problem 2.3
% Output arguments:
% ncc_range - vector of NCC values for the edge lengths from "range"
% ncc_max - location of maximum in NCC

% here you first compute ncc in the given range
bx = 1;  by = 1; % defined start of the search
ncc_range = zeros(1, length(range));

for si = 1:length(range)

    % Cut atomic element
    loc_x_max = bx + range(si); loc_y_max = by + range(si);
    if (loc_x_max <= size(im,2) || loc_y_max <= size(im,1))
        atomic_element = im(bx:loc_x_max - 1, by:loc_y_max - 1);

        % Create periodic image
        nx = ceil(size(im, 2) ./ range(si));
        ny = ceil(size(im, 1) ./ range(si));
        repeat_number = max(nx,ny);
        periodic_large_im = repmat(atomic_element, repeat_number, repeat_number);
        periodic_image = periodic_large_im(1:size(im, 1), 1: size(im,2));
        % Calc NCC   - for full image:
        ncc_range(si) = fcnHandleNCC(im, periodic_image);
    end
end
% then you find maximum in the way you like, just save the numbers you get
%     ncc_max = logical(max(ncc_range));
%     ncc_max_loc = range(ncc_max);
[max_ncc_value, ncc_max_loc] = max(ncc_range);
end

function [ ncc_range , ncc_max_loc ] = compute_cross_correlation_find_max(im , range , fcnHandleNCC )
% Input arguments :
% im - given image (of type double in range 0..1)
% range - vector containing elements from 1 till 300
% fcnHandleNCC - function handle to function " get_ncc (t1 ,t2)" -> see problem 2.3
% Output arguments :
% ncc_range - vector of NCC values for the edge lengths from " range "
% ncc_max - location of maximum in NCC
% here you first compute ncc in the given range
sz = size (im);
bx = 1; by = 1;
ncc_range = zeros(1, max( range ));
for si = 1: length( range )
    s = range(si);
    if (by+s > sz(1) || bx+s > sz(2)); continue ; end
    % Cut atomic element
    atomic = im(by:by+s -1, bx:bx+s -1);
    % Create periodic image
    nr = ceil(max(sz) / s);
    im_per = repmat( atomic , nr , nr);
    im_per = im_per(1: sz(1) , 1: sz(2));
    % Calc NCC - for full image :
    t1 = im_per ; t2 = im;
    ncc_range(s) = fcnHandleNCC(t1 ,t2);
end
% then you find maximum in the way you like , just save the numbers you get
ncc_max_loc= findpeaks( ncc_range , 'MINPEAKHEIGHT', 0.8);
end
