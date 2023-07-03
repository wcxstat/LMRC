alphavec=[1.2,2.0,2.5];
mise_chi1=zeros(5,6);
mise_chi2=zeros(5,6);
mise_pois1=zeros(5,6);
mise_pois2=zeros(5,6);
for err_ind=1:2
for ap_n=1:length(alphavec)
mvec=1:10;
mise_vec1=zeros(1000,length(mvec));
mise_vec2=zeros(1000,length(mvec));
mise_vec3=zeros(1000,length(mvec));
mise_vec4=zeros(1000,length(mvec));
mise_vec5=zeros(1000,length(mvec));

mise_vec1_2=zeros(1000,length(mvec));
mise_vec2_2=zeros(1000,length(mvec));
mise_vec3_2=zeros(1000,length(mvec));
mise_vec4_2=zeros(1000,length(mvec));
mise_vec5_2=zeros(1000,length(mvec));
for rep=1:1000
n=100;
%alpha=1.2;
alpha=alphavec(ap_n);
arg=0:0.01:1;
L=length(arg);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)]; % 50-by-L matrix
beta_coef=4*(-1).^(2:51).*(1:50).^(-3);
beta0=beta_coef*PHI; % true slope function
factor1=sqrt(sum((1:50).^(-alpha).*(beta_coef.^2)));
factor2=sqrt(sum(beta_coef.^2));

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
Y=Xbeta+epsilon;
%Y=exp(Xbeta+sin(Xbeta)-epsilon)+Xbeta-epsilon;
%Y=exp(Xbeta)+0.5*Xbeta.*abs(epsilon)+Xbeta.^3+2*cos(epsilon);
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

[est1,est2]=FLM(Y,kappa,eigfun,xi_est,m);
[~,phi_sir1,phi_sir2]=FSIR(Y,X,arg,m,5);
[~,phi_sir1_2,phi_sir2_2]=FSIR(Y,X,arg,m,10);
[~,phi_cs1,phi_cs2]=FCS(Y,X,arg,m);

inn1=trapz(arg,beta_hat1.*beta0/factor1);
if inn1>=0
    mise_vec1(rep,mi)=trapz(arg,(beta_hat1-beta0/factor1).^2);
    mise_vec1_2(rep,mi)=trapz(arg,(beta_hat2-beta0/factor2).^2);
else
    mise_vec1(rep,mi)=trapz(arg,(-beta_hat1-beta0/factor1).^2);
    mise_vec1_2(rep,mi)=trapz(arg,(-beta_hat2-beta0/factor2).^2);
end

inn2=trapz(arg,est1.*beta0/factor1);
if inn2>=0
    mise_vec2(rep,mi)=trapz(arg,(est1-beta0/factor1).^2);
    mise_vec2_2(rep,mi)=trapz(arg,(est2-beta0/factor2).^2);
else
    mise_vec2(rep,mi)=trapz(arg,(-est1-beta0/factor1).^2);
    mise_vec2_2(rep,mi)=trapz(arg,(-est2-beta0/factor2).^2);
end

inn3=trapz(arg,phi_sir1(:,1)'.*beta0/factor1);
if inn3>=0
    mise_vec3(rep,mi)=trapz(arg,(phi_sir1(:,1)'-beta0/factor1).^2);
    mise_vec3_2(rep,mi)=trapz(arg,(phi_sir2(:,1)'-beta0/factor2).^2);
else
    mise_vec3(rep,mi)=trapz(arg,(-phi_sir1(:,1)'-beta0/factor1).^2);
    mise_vec3_2(rep,mi)=trapz(arg,(-phi_sir2(:,1)'-beta0/factor2).^2);
end

inn4=trapz(arg,phi_sir1_2(:,1)'.*beta0/factor1);
if inn4>=0
    mise_vec4(rep,mi)=trapz(arg,(phi_sir1_2(:,1)'-beta0/factor1).^2);
    mise_vec4_2(rep,mi)=trapz(arg,(phi_sir2_2(:,1)'-beta0/factor2).^2);
else
    mise_vec4(rep,mi)=trapz(arg,(-phi_sir1_2(:,1)'-beta0/factor1).^2);
    mise_vec4_2(rep,mi)=trapz(arg,(-phi_sir2_2(:,1)'-beta0/factor2).^2);
end


inn5=trapz(arg,phi_cs1(:,1)'.*beta0/factor1);
if inn5>=0
    mise_vec5(rep,mi)=trapz(arg,(phi_cs1(:,1)'-beta0/factor1).^2);
    mise_vec5_2(rep,mi)=trapz(arg,(phi_cs2(:,1)'-beta0/factor2).^2);
else
    mise_vec5(rep,mi)=trapz(arg,(-phi_cs1(:,1)'-beta0/factor1).^2);
    mise_vec5_2(rep,mi)=trapz(arg,(-phi_cs2(:,1)'-beta0/factor2).^2);
end
end
end

[min1,ind1]=min(mean(mise_vec1));
[min2,ind2]=min(mean(mise_vec2));
[min3,ind3]=min(mean(mise_vec3));
[min4,ind4]=min(mean(mise_vec4));
[min5,ind5]=min(mean(mise_vec5));
sd1=std(mise_vec1(:,ind1));
sd2=std(mise_vec2(:,ind2));
sd3=std(mise_vec3(:,ind3));
sd4=std(mise_vec4(:,ind4));
sd5=std(mise_vec5(:,ind5));


[min1_2,ind1_2]=min(mean(mise_vec1_2));
[min2_2,ind2_2]=min(mean(mise_vec2_2));
[min3_2,ind3_2]=min(mean(mise_vec3_2));
[min4_2,ind4_2]=min(mean(mise_vec4_2));
[min5_2,ind5_2]=min(mean(mise_vec5_2));
sd1_2=std(mise_vec1_2(:,ind1_2));
sd2_2=std(mise_vec2_2(:,ind2_2));
sd3_2=std(mise_vec3_2(:,ind3_2));
sd4_2=std(mise_vec4_2(:,ind4_2));
sd5_2=std(mise_vec5_2(:,ind5_2));

if err_ind==1
mise_chi1(:,2*ap_n-1)=[min1,min2,min3,min4,min5]';
mise_chi1(:,2*ap_n)=[sd1,sd2,sd3,sd4,sd5]';
mise_chi2(:,2*ap_n-1)=[min1_2,min2_2,min3_2,min4_2,min5_2]';
mise_chi2(:,2*ap_n)=[sd1_2,sd2_2,sd3_2,sd4_2,sd5_2]';
elseif err_ind==2
mise_pois1(:,2*ap_n-1)=[min1,min2,min3,min4,min5]';
mise_pois1(:,2*ap_n)=[sd1,sd2,sd3,sd4,sd5]';
mise_pois2(:,2*ap_n-1)=[min1_2,min2_2,min3_2,min4_2,min5_2]';
mise_pois2(:,2*ap_n)=[sd1_2,sd2_2,sd3_2,sd4_2,sd5_2]';
end
end
end

save('m2_chi1_n100.txt','mise_chi1','-ascii')
save('m2_chi2_n100.txt','mise_chi2','-ascii')
save('m2_pois1_n100.txt','mise_pois1','-ascii')
save('m2_pois2_n100.txt','mise_pois2','-ascii')