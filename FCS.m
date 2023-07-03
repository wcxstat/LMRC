function [lambda,phi1,phi2]=FCS(Y,X,arg,noeig)
% Functional cumulative slicing regression (Yao et al., 2015, Biometrika)

% Input:
% Y: 1*n response
% X: n*length(arg) matrix, dense and regular function data
% noeig: cut-off

n=length(Y); % sample size
ngrid=length(arg); % number of sample point
X_mean=mean(X,1); % sample mean function
X_cen=X-X_mean; %center
xcov=X_cen'*X_cen/n; % covariance
CUM=zeros(ngrid,n);
for i=1:n
    CUM(:,i)=mean(X_cen'.*repmat(double(Y<=Y(i)),ngrid,1),2);
end
CUM_mat=CUM*CUM'/n;

[lambda_x,eigfun_x,~]=FPCA_bal(X,arg,noeig);
phi_x=eigfun_x(:,1:noeig);
xcov_inv=phi_x*diag(lambda_x(1:noeig).^(-1))*phi_x';

h=range(arg)/(ngrid-1);
xcov_inv2=xcov_inv;
xcov_inv2(:,[1,end])=xcov_inv2(:,[1,end])/2;
Gamma=xcov_inv2*CUM_mat*h;

[eigen,d]=eigs(Gamma,ngrid-2,'lm');
d=diag(d); % vectorize
lambda=d(1:noeig)'*h;
phi2=eigen(:,1:noeig);
phi2=phi2/sqrt(h);

phi1=zeros(ngrid,noeig);
for i=1:noeig
    phi2(:,i)=phi2(:,i)/sqrt(trapz(arg,phi2(:,i).^2));
    if phi2(2,i)<phi2(1,i)
       phi2(:,i)=-phi2(:,i);
    end
    delta=phi2(:,i);
    delta([1,end])=phi2([1,end],i)/2;
    phi1(:,i)=phi2(:,i)/sqrt(delta'*xcov*delta*h^2);
end