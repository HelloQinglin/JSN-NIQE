function feat_patch = feature_extract(structdis)
%------------------------------------------------
% Feature Computation
% imdist should be uint8 RGB format
%-------------------------------------------------
data = structdis.data;
if structdis.blockSize(1) == 96
    kernel = [0.500000000000000,0.502768527757491,0.492559208634564,-0.504588182071228;0.500000000000000,-0.498929577944788,0.507373060182250,0.493600905644516;0.500000000000000,0.497222646628556,-0.496379166322284,0.506337202781785;0.500000000000000,-0.501061596441091,-0.503553102494450,-0.495349926355182];
else
    kernel = [0.500000000000000,-0.486482676582460,-0.510091536111064,-0.503131424354505;0.500000000000000,0.516811988724299,-0.489400545845254,0.493348227966011;0.500000000000000,-0.513050205222747,0.479727055395985,0.506597906866961;0.500000000000000,0.482720893080978,0.519765026560387,-0.496814710478482];
end
% feat = [];
feat_patch = zeros(1,38);
impatch = data(:,:,1);
%% A. Statistics of Normalized Luminance
    window = fspecial('gaussian',7,7/6);
    window = window/sum(sum(window));
    
    mu = filter2(window, impatch, 'same');
    mu_sq = mu.*mu;
    sigma = sqrt(abs(filter2(window, impatch.*impatch, 'same') - mu_sq));
    MSCN = (impatch-mu)./(sigma+1);

[alpha, overallstd] = estimateggdparam(MSCN(:));
feat_patch(1:2) = [alpha, overallstd];

%% B. Statistics of MSCN Products
shifts = [0 1;1 0;1 1;1 -1];
for itr_shift =1:4
    shifted_structdis          = circshift(MSCN,shifts(itr_shift,:));
    pair                       = MSCN(:).*shifted_structdis(:);
    [alpha, leftstd, rightstd] = estimateaggdparam(pair);
    const                      =(sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
    meanparam                  =(rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
    start = itr_shift * 4 - 1;
    feat_patch(start:start + 3)                 = [alpha, meanparam, leftstd, rightstd];
end
%% C. PC we extract the Weibull parameters
for index = 2:4
    currentRes = phasecong3(data(:,:,index));
    feat_patch(15 + index * 2:16 + index * 2)  = wblfit(currentRes(:));
end
%% C. PC end

%% MSCN KLT start
    X3 = im2col(MSCN,[2,2],'sliding')';
    coef  = X3*kernel;
    for j = 1:size(coef,2)
        [alpha, overallstd] = estimateggdparam(coef(:,j));
        feat_patch(23+j * 2: 24 + j * 2) = [alpha, overallstd]; 
    end
%% MSCN KLT end

%% GM_LOG start f6
    feat_patch(33:38) = compute_gmlog_features(data(:,:,1)); 
%% GM_LOG end

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
    
%     feat = out([2, 4, 8, 18, 20, 23, 34, 40]);
%     feat = out([2, 4, 18, 23, 34]);
    feat = out([1, 11, 21, 22, 31, 32]);
end
function [gx,gy] = gaussian_derivative(imd,sigma)
    window1 = fspecial('gaussian',2*ceil(3*sigma)+1+2, sigma);
    winx = window1(2:end-1,2:end-1)-window1(2:end-1,3:end);winx = winx/sum(abs(winx(:)));
    winy = window1(2:end-1,2:end-1)-window1(3:end,2:end-1);winy = winy/sum(abs(winy(:)));
    gx = filter2(winx,imd,'same');
    gy = filter2(winy,imd,'same');
end