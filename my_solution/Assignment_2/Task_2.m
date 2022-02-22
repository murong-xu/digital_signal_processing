%% Task2: Zone Plate
%% C2.1
N = 100;
[D, k, Izp] = calculate_zone_plate(N);
figure,imagesc(D);
figure,imshow(Izp);

%% A2.1
Izp21 = Izp(1:2:end, 1:2:end);
Izp41 = Izp(1:4:end, 1:4:end);

figure(1); imshow(Izp);
title('Zone Plate');

figure(2);
imshow(Izp21);
title('Zone Plate 2:1');

figure(3);
imshow(Izp41);
title('Zone Plate 4:1');

% figure(4);
% plot(dc(2:end) - dc(1:end-1));
% title('Argument increase of cosine along diagonal');
%
% imwrite(Izp21, 'res/res_zoneplate_21.png');
% imwrite(Izp41, 'res/res_zoneplate_41.png');

%% C2.1
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