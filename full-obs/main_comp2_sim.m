alphavec=[1.2,2.0,2.5];
mise_chi=zeros(2,6);
mise_pois=zeros(2,6);
for err_ind=1:2
for ap_n=1:length(alphavec)
mvec=1:4;
mise_vec=zeros(1000,length(mvec));
mise_vec_2=zeros(1000,length(mvec));
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
%Y=Xbeta+epsilon;
%Y=exp(Xbeta+sin(Xbeta)-epsilon)+Xbeta-epsilon;
%Y=exp(Xbeta)+0.5*Xbeta.*abs(epsilon)+Xbeta+2*cos(epsilon);
Y=double((1+exp(-Xbeta-epsilon)).^(-1)>0.5);

no_pca=10;
[kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

for mi=1:length(mvec)
m=mvec(mi);
%%%%% Functional SIM %%%%%
[opth,optbeta]=FSIM2(xi_est,Y',m,beta_coef(1:m)');
norm_sim1=sqrt(sum(kappa(1:m).*(optbeta.^2)'));
norm_sim2=sqrt(sum(optbeta.^2));
beta_sim1=optbeta'*eigfun(:,1:m)'/norm_sim1;
beta_sim2=optbeta'*eigfun(:,1:m)'/norm_sim2;
%%%%%%%%%%%%%%%%%%%%
inn=trapz(arg,beta_sim1.*beta0/factor1);
if inn>=0
    mise_vec(rep,mi)=trapz(arg,(beta_sim1-beta0/factor1).^2);
    mise_vec_2(rep,mi)=trapz(arg,(beta_sim2-beta0/factor2).^2);
else
    mise_vec(rep,mi)=trapz(arg,(-beta_sim1-beta0/factor1).^2);
    mise_vec_2(rep,mi)=trapz(arg,(-beta_sim2-beta0/factor2).^2);
end
end
end

[min1,ind1]=min(mean(mise_vec));
sd1=std(mise_vec(:,ind1));


[min1_2,ind1_2]=min(mean(mise_vec_2));
sd1_2=std(mise_vec_2(:,ind1_2));

if err_ind==1
mise_chi(1,2*ap_n-1)=min1;
mise_chi(1,2*ap_n)=sd1;
mise_chi(2,2*ap_n-1)=min1_2;
mise_chi(2,2*ap_n)=sd1_2;
elseif err_ind==2
mise_pois(1,2*ap_n-1)=min1;
mise_pois(1,2*ap_n)=sd1;
mise_pois(2,2*ap_n-1)=min1_2;
mise_pois(2,2*ap_n)=sd1_2;
end
end
end