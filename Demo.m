clear;clc;close all;

templateModel = load('MVG_model');
templateModel = templateModel.MVGModel;
mu_prisparam = templateModel{1};
cov_prisparam = templateModel{2};

%% load image names
img2 = imread('img2.bmp');
quality = computequality(img2,mu_prisparam,cov_prisparam);