mvec=1:15;
nvec=500:200:3500;
alphavec=[1.2,2.0,2.5];
mise_chi1=zeros(length(alphavec),length(nvec));
mise_chi2=zeros(length(alphavec),length(nvec));
mise_pois1=zeros(length(alphavec),length(nvec));
mise_pois2=zeros(length(alphavec),length(nvec));
for err_ind=1:2
for ap_n=1:length(alphavec)
%mise=zeros(1,length(nvec));
for nn=1:length(nvec)
mise_vec1=zeros(1000,length(mvec));
mise_vec2=zeros(1000,length(mvec));
n=nvec(nn);
alpha=alphavec(ap_n);
arg=0:0.01:1;
L=length(arg);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)]; % 50-by-L matrix
beta_coef=4*(-1).^(2:51).*(1:50).^(-3);
beta0=beta_coef*PHI; % true slope function
factor1=sqrt(sum((1:50).^(-alpha).*(beta_coef.^2)));
factor2=sqrt(sum(beta_coef.^2));

for rep=1:1000
Z=normrnd(0,1,n,50);
eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
score=Z.*repmat(eigvalsq,n,1);
X=score*PHI;
Xbeta=(beta_coef.*eigvalsq)*Z';
if err_ind==1
epsilon=chi2rnd(1,[1,n]);
elseif err_ind==2
epsilon=random('Poisson',1,1,n);
end
%Y=Xbeta+epsilon;
Y=exp(Xbeta+sin(Xbeta)-epsilon)+Xbeta-epsilon;
%Y=exp(Xbeta)+0.5*Xbeta.*abs(epsilon)+Xbeta+2*cos(epsilon);
%Y=double((1+exp(-Xbeta-epsilon)).^(-1)>0.5);

no_pca=30;
[kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);
gvec=zeros(1,no_pca);
for i=1:n
    Y_cp=Y;Y_cp(i)=[];
    xi_cp=xi_est;xi_cp(i,:)=[];
    diff=double(Y(i)<Y_cp);
    gvec=gvec+mean(repmat(diff',1,no_pca).*(xi_cp-xi_est(i,:)),1);
end
gvec=gvec/n;

for mi=1:length(mvec)
m=mvec(mi);
norm_factor1=sqrt(sum(kappa(1:m).^(-1).*(gvec(1:m).^2)));
norm_factor2=sqrt(sum(kappa(1:m).^(-2).*(gvec(1:m).^2)));
bcoef_hat1=kappa(1:m).^(-1).*gvec(1:m)/norm_factor1;
bcoef_hat2=kappa(1:m).^(-1).*gvec(1:m)/norm_factor2;
beta_hat1=bcoef_hat1*eigfun(:,1:m)';
beta_hat2=bcoef_hat2*eigfun(:,1:m)';
mise_vec1(rep,mi)=trapz(arg,(beta_hat1-beta0/factor1).^2);
mise_vec2(rep,mi)=trapz(arg,(beta_hat2-beta0/factor2).^2);
end
end
if err_ind==1
mise_chi1(ap_n,nn)=min(mean(mise_vec1));
mise_chi2(ap_n,nn)=min(mean(mise_vec2));
elseif err_ind==2
mise_pois1(ap_n,nn)=min(mean(mise_vec1));
mise_pois2(ap_n,nn)=min(mean(mise_vec2));
end
end
end
end

save('mise_m1_chi1.txt','mise_chi1','-ascii')
save('mise_m1_chi2.txt','mise_chi2','-ascii')
save('mise_m1_pois1.txt','mise_pois1','-ascii')
save('mise_m1_pois2.txt','mise_pois2','-ascii')

plot(log(nvec),log(mise),'o-')
hold on
plot(log(nvec),-(2*3-1)/(alpha+2*3)*log(nvec))