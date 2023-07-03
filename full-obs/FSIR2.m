function [lambda,phi1,phi2]=FSIR2(Y,X,arg,noeig,S)
% functional sliced inverse regression (Ferre & Yao, 2005)

% Input:
% Y: 1*n response
% X: n*length(arg) matrix, dense and regular function data
% noeig: cut-off
% S: the number of slices

n=length(Y); % sample size
ngrid=length(arg); % number of sample point
X_mean=mean(X,1); % sample mean function
X_cen=X-X_mean; %center
xcov=X_cen'*X_cen/n; % covariance

point=linspace(min(Y),max(Y),S+1);
SImat=zeros(ngrid,ngrid);
for s=1:S
    if s==S
        index=point(s)<=Y & Y<=point(s+1);
    else
        index=point(s)<=Y & Y<point(s+1);
    end
    pr=sum(index)/n;
    if sum(index)>=1
        smean=mean(X(index,:),1)-X_mean;
        SImat=SImat+smean'*smean*pr;
    end
end

[lambda_x,eigfun_x,~]=FPCA_bal(X,arg,noeig);
phi_x=eigfun_x(:,1:noeig);
xcov_inv_sq=phi_x*diag(lambda_x(1:noeig).^(-1/2))*phi_x';

h=range(arg)/(ngrid-1);
%xcov_inv_sq2=xcov_inv_sq;
%xcov_inv_sq2(:,[1,end])=xcov_inv_sq2(:,[1,end])/2;
%Gamma=xcov_inv2*SImat*h;
Gamma=xcov_inv_sq*SImat*xcov_inv_sq*h^2;

[eigen,d]=eigs(Gamma,ngrid-2,'lm');
d=diag(d); % vectorize
lambda=d(1:noeig)'*h;
phi2=xcov_inv_sq*eigen(:,1:noeig);
%phi2=phi2/sqrt(h);

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