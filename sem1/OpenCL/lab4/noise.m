I = imread('SobelFilterImage_Input.bmp');
I_noise_gray = imnoise(rgb2gray(I),'salt & pepper',0.10);
I_med = medfilt2(I_noise_gray, [3 3] , 'symmetric');
L=cat(3,I_noise_gray,I_noise_gray,I_noise_gray)
imwrite(L, 'median_3x3_matlab.bmp');
imshow(I_med)