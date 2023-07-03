function [est1,est2]=FLM_sparse(g_vec,kappa,eigfun,m)

norm_factor1=sqrt(sum(kappa(1:m).^(-1).*(g_vec(1:m).^2)));
norm_factor2=sqrt(sum(kappa(1:m).^(-2).*(g_vec(1:m).^2)));
bcoef_hat1=kappa(1:m).^(-1).*g_vec(1:m)/norm_factor1;
bcoef_hat2=kappa(1:m).^(-1).*g_vec(1:m)/norm_factor2;
est1=bcoef_hat1*eigfun(:,1:m)';
est2=bcoef_hat2*eigfun(:,1:m)';