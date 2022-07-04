function quality = computequality(imDist,mu_prisparam,cov_prisparam)
%%

blocksize_all    = 96;
blocksizerow    = 96;
blocksizecol    = 96;

%% 
infConst = 10000;
nanConst = 0;
%%
% imDist = imresize(imDist, [normalizedWidth normalizedWidth]);
[row, col, ~]    = size(imDist);
block_rownum     = floor(row/blocksize_all);
block_colnum     = floor(col/blocksize_all);
imDist           = imDist(1:block_rownum*blocksize_all,1:block_colnum*blocksize_all,:);

feat = [];
featnum = 38;
% index = logical([zeros(1,18) ones(1,6) zeros(1,8) ones(1,6)]);
for itr_scale = 1:2
    imDist = imresize(imDist, 1/itr_scale);
    impatch = double(rgb2gray(imDist));
    blocksize = blocksize_all / itr_scale;
    %% prepare
%     window = fspecial('gaussian',7,7/6);
%     window = window/sum(sum(window));
%     
%     mu = filter2(window, impatch, 'same');
%     mu_sq = mu.*mu;
%     sigma = sqrt(abs(filter2(window, impatch.*impatch, 'same') - mu_sq));
%     structdis = (impatch-mu)./(sigma+1);
    
    %% prepare PC
    impatch_color_d = double(imDist);
    O1 = 0.3*impatch_color_d(:,:,1) + 0.04*impatch_color_d(:,:,2) - 0.35*impatch_color_d(:,:,3);
    O2 = 0.34*impatch_color_d(:,:,1) - 0.6*impatch_color_d(:,:,2) + 0.17*impatch_color_d(:,:,3);
    O3 = 0.06*impatch_color_d(:,:,1) + 0.63*impatch_color_d(:,:,2) + 0.27*impatch_color_d(:,:,3);
%     PCO1 = phasecong3(O1);
%     PCO2 = phasecong3(O2);
%     PCO3 = phasecong3(O3);

    
    compositeMat = zeros(size(impatch,1),size(impatch,2),4);
    compositeMat(:,:,1) = impatch;
    compositeMat(:,:,2) = O1;
    compositeMat(:,:,3) = O2;
    compositeMat(:,:,4) = O3;
%     compositeMat(:,:,5) = impatch;
    %% calculate
    feat_scale = blockproc(compositeMat,[blocksizerow/itr_scale blocksizecol/itr_scale],@feature_extract,...
                               'UseParallel',1,'TrimBorder',0);
%     bb = reshape(feat_scale, block_rownum * 38,[]);
%     feat_ni = feat_scale';
%     bb = reshape(feat_ni, 38,[]);
%     feat_scale = bb';
    feat_scale = reshape(feat_scale,featnum * block_rownum, block_colnum);
    feat_now = zeros(block_rownum * block_colnum, featnum);
    for i = 1 : 38
        feat_now(:, i) = reshape(feat_scale(i * block_rownum - block_rownum + 1: i * block_rownum, :),[], 1);
    end
    feat = [feat feat_now];
%     feat = [feat feat_now(:,~index)]; % select NSS feature
end
%% 注释掉
% % imDist = double(rgb2gray(imDist));
% 
% % feat_all=brisque_feature(imDist);
% % [feat, ~]  = feature_extract_0303(imDist, blocksize,kltsize,1);
% [feat, ~]  = feature_extract_0505_v2(imDist, blocksize,kernel1);
% imDist = imresize(imDist, 0.5);
% % [feat2, ~] = feature_extract_0303(imDist, blocksize/2,kltsize,2);
% [feat2, ~] = feature_extract_0505_v2(imDist, blocksize/2,kernel2);
% feat = [feat, feat2];

%% 异常值处理
infIndicator = isinf(feat);
feat(infIndicator) = infConst;
nanIndicator = isnan(feat);
feat(nanIndicator) = nanConst;
%%
mu_distparam       = nanmean(feat);
cov_distparam      = nancov(feat);

nanIndicator = isnan(mu_distparam);
mu_distparam(nanIndicator) = nanConst;
nanIndicator = isnan(cov_distparam);
cov_distparam(nanIndicator) = nanConst;

invcov_param = pinv((cov_prisparam+cov_distparam)/2);

quality = sqrt((mu_prisparam-mu_distparam)* ...
    invcov_param*(mu_prisparam-mu_distparam)');

quality = real(quality);
end