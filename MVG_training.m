clear;clc;close all;warning off;

im_pth = 'E:\Database\pristinedata\pristine\';

blocksize    = 96;
sh_th = 0.75;

X_Y  = [];
%%
current = pwd;
cd(sprintf('%s',im_pth))
names        = ls;
names        = names(3:end,:);
cd(current);

%%
for idx_im = 1:size(names,1)
    tic
    idx_im
    img_org = imread([im_pth,names(idx_im,:)]);
    kltsize = 2;
    [feat, sharpness] = feature_extract_training(img_org, blocksize, kltsize);
    img_org = imresize(img_org, 0.5);
    [feat2, ~] = feature_extract_training(img_org, blocksize/2, kltsize);
    %%
    IX = find(sharpness(:) >sh_th*max(sharpness(:)));
    X_Y  = [X_Y; feat(IX,:) feat2(IX,:)];

    toc
end

%% MVG Training

mu_prisparam       = nanmean(X_Y);
cov_prisparam      = nancov(X_Y);

MVGModel{1} = mu_prisparam;
MVGModel{2} = cov_prisparam;

save('MVG_model.mat','-v7.3','MVGModel');






