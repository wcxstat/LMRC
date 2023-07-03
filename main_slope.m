mvec=1:4;
arg=0:0.01:1;
L=length(arg);
n=500;
alphavec=[1.2,2.0,2.5];
beta_chi1=zeros(4*length(alphavec),L);
beta_chi2=zeros(4*length(alphavec),L);
beta_pois1=zeros(4*length(alphavec),L);
beta_pois2=zeros(4*length(alphavec),L);
for err_ind=1:2
for ap_n=1:length(alphavec)
mise_vec1=zeros(1000,length(mvec));
mise_vec2=zeros(1000,length(mvec));
beta_mat=zeros(1000,length(mvec)*L);
beta_mat2=zeros(1000,length(mvec)*L);
alpha=alphavec(ap_n);
fprintf('error=%d, ap_n=%d\n',err_ind,alpha);

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
%Y=exp(Xbeta+sin(Xbeta)-epsilon)+Xbeta-epsilon;
%Y=exp(Xbeta)+0.5*Xbeta.*abs(epsilon)+Xbeta.^3+2*cos(epsilon);
Y=double((1+exp(-Xbeta-epsilon)).^(-1)>0.5);

no_pca=10;
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

index=((mi-1)*L+1):(mi*L);
beta_mat(rep,index)=beta_hat1;
beta_mat2(rep,index)=beta_hat2;
end
end

[~,ind]=min(mean(mise_vec1));
[~,ind2]=min(mean(mise_vec2));
index_opt=((ind-1)*L+1):(ind*L);
index_opt2=((ind2-1)*L+1):(ind2*L);
beta_opt=beta_mat(:,index_opt);
beta_opt2=beta_mat2(:,index_opt2);

est1=[beta0/factor1;mean(beta_opt);
    quantile(beta_opt,[0.05,0.95],1)];
est2=[beta0/factor2;mean(beta_opt2);
    quantile(beta_opt2,[0.05,0.95],1)];

ind3=((ap_n-1)*4+1):(ap_n*4);
if err_ind==1
beta_chi1(ind3,:)=est1;
beta_chi2(ind3,:)=est2;
elseif err_ind==2
beta_pois1(ind3,:)=est1;
beta_pois2(ind3,:)=est2;
end
end
end

save('beta_m4_chi1_n500.txt','beta_chi1','-ascii')
save('beta_m4_chi2_n500.txt','beta_chi2','-ascii')
save('beta_m4_pois1_n500.txt','beta_pois1','-ascii')
save('beta_m4_pois2_n500.txt','beta_pois2','-ascii')


plot(arg,beta0/factor1)
hold on
plot(arg,mean(beta_opt))
plot(arg,prctile(beta_opt,5))
plot(arg,prctile(beta_opt,95))
plot(arg,prctile(beta_opt,50))