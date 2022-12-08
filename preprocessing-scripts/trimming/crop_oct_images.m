clc
clear all
close all

img=imread('NORMAL-1384-12.jpeg');


threshold =80;  %%%% is computed manually by analysing the intensity of the RPE  layer

retina = NaN(1, size(img,2));

for j = 1:size(img,2)
    retina(j) = find(img(:,j) >= threshold, 1, 'last');
end

% Fit polynomial
x = 1:length(retina);
p = polyfit(x, retina, 2);
retina_fit = round(polyval(p, x));

cropped_img = zeros(size(img), 'uint8');

%%%----- the crop top and bottom are also defined by analysing the images
%%%in the dataset.
crop_top=120;
crop_bottom=40;


cropped_img = img(max(retina_fit)-crop_top:max(retina_fit)+crop_bottom,:);

figure;
imshow(img,[]);
hold on;
plot(x, retina_fit);
plot(x, retina)
figure;
imshow(cropped_img,[])