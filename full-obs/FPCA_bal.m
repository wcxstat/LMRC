function [lambda, eigfun, xi_est] = FPCA_bal(x, t, noeig)
% a simple code to carry out functional principal component analysis
% based on regular and dense functional data without any measurement error
% 3-10 2017 by Wenchao Xu

% Input
% x :     a n times length(t) matrix, dense and regular function data
% t :     a vector
% noeig : number of functional PCA score needed

% Output
% lambda : eigenvalue
% eigen :  eigenfunction, length(t)-by-noeig matrix
% xi :     functional principal component score, n tims noeig matrix

n = size(x,1); % sample size
ngrid = length(t); % number of sample point, x is a n-by-ngrid matrix
x_mean = mean(x,1); % sample mean function
x_cen = x - repmat(x_mean,n,1); %center
xcov = x_cen' * x_cen/n; % sample covariance function, ngrid-by-ngrid matrix
h = range(t)/(ngrid - 1);
[eigen, d] = eigs(xcov,ngrid - 2,'lm');
d = diag(d); % vectorize
lambda = d(1:noeig)' * h;
eigfun = eigen(:,1:noeig);
eigfun = eigfun / sqrt(h);
for i = 1: noeig
    eigfun(:,i) = eigfun(:,i)/sqrt(trapz(t,eigfun(:,i).^2));
    if eigfun(2,i) < eigfun(1,i)
       eigfun(:,i) = -eigfun(:,i);
    end
end
xi_est = zeros(n, noeig);
for j = 1:noeig
    xi_est(:,j) = trapz(t, (x_cen .* repmat(eigfun(:,j)',n,1))');
end