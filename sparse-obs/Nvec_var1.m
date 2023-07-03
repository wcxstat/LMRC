addpath(genpath('/Users/xuwenchao/Documents/PACE_matlab'));
newt=linspace(0,1,101);
opt=6;
p=setOptions('regular',0,'error',1,'selection_k',opt,'kernel','gauss',...
    'newdata',newt);

mvec=1:4;
n=100;
Jmax=8;
Nvec=[5,10,20,30,100]; % 100 is dense FDA
alphavec=[1.1,1.5,1.8];
L=length(newt);
PHI=[ones(1,L);sqrt(2)*cos((1:(Jmax-1))'*newt*pi)]; % Jamx-by-L matrix
beta_coef=4*(-1).^(2:(Jmax+1)).*(1:Jmax).^(-3);
beta0=beta_coef*PHI; % true slope function
factor1=5*sqrt(sum((1:Jmax).^(-alpha).*(beta_coef.^2)));
factor2=sqrt(sum(beta_coef.^2));

mise_chi=zeros(100,length(Nvec));
mise_chi2=zeros(100,length(Nvec));
mise_pois=zeros(100,length(Nvec));
mise_pois2=zeros(100,length(Nvec));
for err_ind=1:2
for ap_n=1:length(alphavec)
alpha=alphavec(ap_n);
for nn=1:length(Nvec)
mise_vec=zeros(100,length(mvec));
mise_vec2=zeros(100,length(mvec));
Ni=Nvec(nn);
for rep=1:100
fprintf('error=%d, alpha=%d, N=%d, and repeat=%d\n',err_ind,alpha,Ni,rep)
if err_ind==1
epsilon=chi2rnd(1,[1,n]);
elseif err_ind==2
epsilon=random('Poisson',1,1,n);
end
Z=normrnd(0,1,n,Jmax);
eigvalsq=5*(-1).^(2:(Jmax+1)).*(1:Jmax).^(-alpha/2);

if Ni<=50
t=cell(1,n);
u=cell(1,n);
Y=zeros(1,n);
for i=1:n
    t{i}=sort(unifrnd(0,1,[1,Ni]));
    score=Z(i,:).*eigvalsq;
    phi=[ones(Ni,1),sqrt(2)*cos(pi*t{i}'*(1:(Jmax-1)))];
    u{i}=score*phi'+normrnd(0,sqrt(0.1),1);
    Xbeta=(beta_coef.*eigvalsq)*Z(i,:)';
    %Y(i)=Xbeta+epsilon(i);
    Y(i)=exp(Xbeta+sin(Xbeta)-epsilon(i))+Xbeta-epsilon(i);
    %Y(i)=exp(Xbeta)+0.5*Xbeta.*abs(epsilon(i))+Xbeta.^3+2*cos(epsilon(i));
    %Y(i)=double((1+exp(-Xbeta-epsilon(i)))^(-1)>0.5);
end

xx=FPCA(u,t,p);
kappa=getVal(xx,'lambda');
eigfun=getVal(xx,'phi');
%mu=getVal(xx,'mu');

wu=cell(1,n);
for i=1:n
    Y_cp=Y;Y_cp(i)=[];
    wsum=(sum(Y_cp<Y(i))-sum(Y(i)<Y_cp))/(n-1);
    wu{i}=wsum*u{i};
end
Tvec=cell2mat(t);
wuvec=cell2mat(wu);
[bw,gcv]=gcv_lwls(wuvec,Tvec,'gauss',-2,1,0,2,'off',0);
[invalid,g]=lwls(bw,'gauss',-2,1,0,Tvec,wuvec',...
    ones(1,length(Tvec)),newt,0);
gvec=trapz(newt,eigfun.*repmat(g',1,opt));
else
score=Z.*repmat(eigvalsq,n,1);
X=score*PHI;
Xbeta=(beta_coef.*eigvalsq)*Z';
%Y=Xbeta+epsilon;
Y=exp(Xbeta+sin(Xbeta)-epsilon)+Xbeta-epsilon;
%Y=exp(Xbeta)+0.5*Xbeta.*abs(epsilon)+Xbeta+2*cos(epsilon);
%Y=double((1+exp(-Xbeta-epsilon)).^(-1)>0.5);

no_pca=7;
[kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);
gvec=zeros(1,no_pca);
for i=1:n
    Y_cp=Y;Y_cp(i)=[];
    xi_cp=xi_est;xi_cp(i,:)=[];
    diff=double(Y(i)<Y_cp);
    gvec=gvec+mean(repmat(diff',1,no_pca).*(xi_cp-xi_est(i,:)),1);
end
gvec=gvec/n;
end

for mi=1:length(mvec)
    m=mvec(mi);
    norm_factor1=sqrt(sum(kappa(1:m).^(-1).*(gvec(1:m).^2)));
    norm_factor2=sqrt(sum(kappa(1:m).^(-2).*(gvec(1:m).^2)));
    bcoef_hat1=kappa(1:m).^(-1).*gvec(1:m)/norm_factor1;
    bcoef_hat2=kappa(1:m).^(-1).*gvec(1:m)/norm_factor2;
    beta_hat1=bcoef_hat1*eigfun(:,1:m)';
    beta_hat2=bcoef_hat2*eigfun(:,1:m)';

    mise_vec(rep,mi)=trapz(newt,(beta_hat1-beta0/factor1).^2);
    mise_vec2(rep,mi)=trapz(newt,(beta_hat2-beta0/factor2).^2); 
end
end

[~,ind]=min(mean(mise_vec));
[~,ind2]=min(mean(mise_vec2));
index=(ap_n-1)*length(Nvec)+nn;
if err_ind==1
mise_chi(:,index)=mise_vec(:,ind);
mise_chi2(:,index)=mise_vec2(:,ind2);
elseif err_ind==2
mise_pois(:,index)=mise_vec(:,ind);
mise_pois2(:,index)=mise_vec2(:,ind2);
end
end
end
end

save('nc_m2_chi1_n100.txt','mise_chi','-ascii')
save('nc_m2_chi2_n100.txt','mise_chi2','-ascii')
save('nc_m2_pois1_n100.txt','mise_pois','-ascii')
save('nc_m2_pois2_n100.txt','mise_pois2','-ascii')