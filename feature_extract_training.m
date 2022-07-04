function [feat, sharpness] = feature_extract_training(imdist, blocksize, kltsize)
%------------------------------------------------
% Feature Computation
% imdist should be uint8 RGB format
%-------------------------------------------------
[row, col, ~] = size(imdist);
block_rownum  = floor(row/blocksize);
block_colnum  = floor(col/blocksize);
imdist        = imdist(1:block_rownum*blocksize,1:block_colnum*blocksize,:);
%%
feat         = zeros(block_rownum*block_colnum, 38);
sharpness    = zeros(block_rownum*block_colnum,1);
%% block level feature extraction
blk{1} = im2col(imdist(:,:,1),[blocksize blocksize],'distinct')';
blk{2} = im2col(imdist(:,:,2),[blocksize blocksize],'distinct')';
blk{3} = im2col(imdist(:,:,3),[blocksize blocksize],'distinct')';
%%
for i = 1:block_rownum*block_colnum
    feat_patch = zeros(1,38);
    impatch_color(:,:,1) = reshape(blk{1}(i,:),[blocksize blocksize]);
    impatch_color(:,:,2) = reshape(blk{2}(i,:),[blocksize blocksize]);
    impatch_color(:,:,3) = reshape(blk{3}(i,:),[blocksize blocksize]);

    impatch = double( rgb2gray(impatch_color) );
    %% MSCN
    window = fspecial('gaussian',7,7/6);
    window = window/sum(sum(window));
    
    mu = filter2(window, impatch, 'same');
    mu_sq = mu.*mu;
    sigma = sqrt(abs(filter2(window, impatch.*impatch, 'same') - mu_sq));
    structdis = (impatch-mu)./(sigma+1);
    
    % sharpness
    sharpness(i,:) = mean(sigma(:));

    [alpha, overallstd] = estimateggdparam(structdis(:));
    feat_patch(1:2) = [alpha, overallstd];
    
    shifts = [0 1;1 0 ;1 1;1 -1];
    for itr_shift =1:4
        shifted_structdis          = circshift(structdis,shifts(itr_shift,:));
        pair                       = structdis(:).*shifted_structdis(:);
        [alpha, leftstd, rightstd] = estimateaggdparam(pair);
        const                      =(sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
        meanparam                  =(rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
        start = itr_shift * 4 - 1;
        feat_patch(start:start + 3)                 = [alpha, meanparam, leftstd, rightstd];
    end
    %% PC Color
    impatch_color_d = double(impatch_color);
    O1 = 0.3*impatch_color_d(:,:,1) + 0.04*impatch_color_d(:,:,2) - 0.35*impatch_color_d(:,:,3);
    O2 = 0.34*impatch_color_d(:,:,1) - 0.6*impatch_color_d(:,:,2) + 0.17*impatch_color_d(:,:,3);
    O3 = 0.06*impatch_color_d(:,:,1) + 0.63*impatch_color_d(:,:,2) + 0.27*impatch_color_d(:,:,3);
    structdisPC = phasecong3(O1);
    feat_patch(19:20)  = wblfit(structdisPC(:));
    structdisPC = phasecong3(O2);
    feat_patch(21:22)  = wblfit(structdisPC(:));
    structdisPC = phasecong3(O3);
    feat_patch(23:24)  = wblfit(structdisPC(:));
    %% MSCN KLT
    if blocksize==96
        load(['kernel/kernel_MSCN_x',num2str(kltsize),'_gray']);
    elseif blocksize==48
        load(['kernel/kernel_MSCN_ds2_x',num2str(kltsize),'_gray']);
    end

    X3 = im2col(structdis,[2,2],'sliding')';
    coef  = X3*kernel{1};
    for j = 1:size(coef,2)
        [alpha, overallstd] = estimateggdparam(coef(:,j));
        feat_patch(23+j * 2: 24 + j * 2) = [alpha, overallstd]; 
    end

    %% GM-LOG vector = 6 each scale
    feat_patch(33:38) = compute_gmlog_features(impatch); 
    %% feature fusion
    feat(i, :) = feat_patch;
end

end

%% compute_gmlog_features
function feat = compute_gmlog_features(imgray)
% compute GM-LOG features # 2(std), 4(std), 8, 18(std), 20, 23, 34(std), 40(both) 
    sigma = 0.5;
    [gx,gy] = gaussian_derivative(imgray,sigma);
    grad_im = sqrt(gx.^2+gy.^2);

    window2 = fspecial('log', 2*ceil(3*sigma)+1, sigma);%Laplacian of Gaussian filter
    window2 =  window2/sum(abs(window2(:)));
    log_im = abs(filter2(window2, imgray, 'same'));

    ratio = 2.5; % default value 2.5 is the average ratio of GM to LOG on LIVE database
    grad_im = abs(grad_im/ratio);

    %Normalization
    c0 = 4*0.05;
    sigmaN = 2*sigma;
    window1 = fspecial('gaussian',2*ceil(3*sigmaN)+1, sigmaN);
    window1 = window1/sum(window1(:));
    Nmap = sqrt(filter2(window1,mean(cat(3,grad_im,log_im).^2,3),'same'))+c0;
    grad_im = (grad_im)./Nmap;
    log_im = (log_im)./Nmap;
    % remove the borders, which may be the wrong results of a convolution
    % operation
    h = ceil(3*sigmaN);
    grad_im = abs(grad_im(h:end-h+1,h:end-h+1,:));
    log_im = abs(log_im(h:end-h+1,h:end-h+1));

    ctrs{1} = 1:10;ctrs{2} = 1:10;
    % histogram computation
    step1 = 0.20;
    step2 = 0.20;
    grad_qun = ceil(grad_im/step1);
    log_im_qun = ceil(log_im/step2);

    N1 = hist3([grad_qun(:),log_im_qun(:)],ctrs);
    N1 = N1/sum(N1(:));
    NG = sum(N1,2);%行求和
    NL = sum(N1,1);%列求和

    alpha1 = 0.0001;
    % condition probability: Grad conditioned on LOG
    cp_GL = N1./(repmat(NL,size(N1,1),1)+alpha1);
    cp_GL_H=  sum(cp_GL,2)';
    cp_GL_H = cp_GL_H/sum(cp_GL_H);
    % condition probability: LOG conditioned on Grad
    cp_LG = N1./(repmat(NG,1,size(N1,2))+alpha1);
    cp_LG_H = sum(cp_LG,1);
    cp_LG_H = cp_LG_H/(sum(cp_LG_H));

    out = [NG', NL, cp_GL_H,cp_LG_H];
    feat = out([1, 11, 21, 22, 31, 32]);
end
function [gx,gy] = gaussian_derivative(imd,sigma)
    window1 = fspecial('gaussian',2*ceil(3*sigma)+1+2, sigma);
    winx = window1(2:end-1,2:end-1)-window1(2:end-1,3:end);winx = winx/sum(abs(winx(:)));
    winy = window1(2:end-1,2:end-1)-window1(3:end,2:end-1);winy = winy/sum(abs(winy(:)));
    gx = filter2(winx,imd,'same');
    gy = filter2(winy,imd,'same');
end