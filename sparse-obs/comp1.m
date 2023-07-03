addpath(genpath('E:\xuwenchao\MAFLQR\PACE_matlab'));
newt=linspace(0,1,101);
opt=6;
p=setOptions('regular',0,'error',1,'selection_k',opt,'kernel','gauss',...
    'newdata',newt);

mvec=1:4;
n=100;
Jmax=8;
%alpha=1.8;
L=length(newt);
PHI=[ones(1,L);sqrt(2)*cos((1:(Jmax-1))'*newt*pi)]; % Jamx-by-L matrix
beta_coef=4*(-1).^(2:(Jmax+1)).*(1:Jmax).^(-3);
beta0=beta_coef*PHI; % true slope function

alphavec=[1.1,1.5,1.8];
mise_chi1=zeros(3,6);
mise_chi2=zeros(3,6);
mise_pois1=zeros(3,6);
mise_pois2=zeros(3,6);
for err_ind=1:2
for ap_n=1:length(alphavec)
    alpha=alphavec(ap_n);
factor1=5*sqrt(sum((1:Jmax).^(-alpha).*(beta_coef.^2)));
factor2=sqrt(sum(beta_coef.^2));

mise_vec1=zeros(100,length(mvec));
mise_vec2=zeros(100,length(mvec));
mise_vec3=zeros(100,length(mvec));
mise_vec1_2=zeros(100,length(mvec));
mise_vec2_2=zeros(100,length(mvec));
mise_vec3_2=zeros(100,length(mvec));
for rep=1:100
if err_ind==1
epsilon=chi2rnd(1,[1,n]);
elseif err_ind==2
epsilon=random('Poisson',1,1,n);
end
Z=normrnd(0,1,n,Jmax);
N=unidrnd(6,[1,n])+4;
t=cell(1,n);
u=cell(1,n);
Y=zeros(1,n);
for i=1:n
    Ni=N(i);
    t{i}=sort(unifrnd(0,1,[1,Ni]));
    eigvalsq=5*(-1).^(2:(Jmax+1)).*(1:Jmax).^(-alpha/2);
    score=Z(i,:).*eigvalsq;
    phi=[ones(Ni,1),sqrt(2)*cos(pi*t{i}'*(1:(Jmax-1)))];
    u{i}=score*phi'+normrnd(0,sqrt(0.1),1);
    Xbeta=(beta_coef.*eigvalsq)*Z(i,:)';
    Y(i)=Xbeta+epsilon(i);
    %Y(i)=exp(Xbeta+sin(Xbeta)-epsilon(i))+Xbeta-epsilon(i);
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

%%%%%%%% FLM %%%%%%%%
yx=cell(1,n);
for i=1:n
    yx{i}=(Y(i)-mean(Y))*u{i};
end
yxvec=cell2mat(yx);
[bw,~]=gcv_lwls(yxvec,Tvec,'gauss',-2,1,0,2,'off',0);
[~,gflm]=lwls(bw,'gauss',-2,1,0,Tvec,yxvec',...
    ones(1,length(Tvec)),newt,0);
gflm_vec=trapz(newt,eigfun.*repmat(gflm',1,opt));
%%%%%%%%%%%%%%%%

%%%%%%%% FCS %%%%%%%%
mu=getVal(xx,'mu');
xcov=eigfun*diag(kappa)*eigfun';
muvec=interp1(newt,mu,Tvec,'spline');
mreg_mat=zeros(L,n);
for i=1:n
    YCS=double(Y<=Y(i));
    YCS_vec=ones(1,length(Tvec));
    ia=0;
    for j=1:n
        index=(ia+1):(ia+N(j));
        YCS_vec(index)=repmat(YCS(j),1,N(j));
        ia=ia+N(j);
    end
    uy_vec=YCS_vec.*(cell2mat(u)-muvec);
    [bw,~]=gcv_lwls(uy_vec,Tvec,'gauss',-2,1,0,2,'off',0);
    [~,mhat]=lwls(bw,'gauss',-2,1,0,Tvec,uy_vec',...
        ones(1,length(Tvec)),newt,0);
    mreg_mat(:,i)=mhat';
end
kerfun=mreg_mat*mreg_mat'/n;
%%%%%%%%%%%%%%%%


for mi=1:length(mvec)
    m=mvec(mi);
    norm_factor1=sqrt(sum(kappa(1:m).^(-1).*(gvec(1:m).^2)));
    norm_factor2=sqrt(sum(kappa(1:m).^(-2).*(gvec(1:m).^2)));
    bcoef_hat1=kappa(1:m).^(-1).*gvec(1:m)/norm_factor1;
    bcoef_hat2=kappa(1:m).^(-1).*gvec(1:m)/norm_factor2;
    beta_hat1=bcoef_hat1*eigfun(:,1:m)';
    beta_hat2=bcoef_hat2*eigfun(:,1:m)';
    
    [est1,est2]=FLM_sparse(gflm_vec,kappa,eigfun,m);
    [~,phi_cs1,phi_cs2]=FCS_sparse(kerfun,xcov,kappa,eigfun,newt,m);

    inn1=trapz(newt,beta_hat1.*beta0/factor1);
    if inn1>=0
      mise_vec1(rep,mi)=trapz(newt,(beta_hat1-beta0/factor1).^2);
      mise_vec1_2(rep,mi)=trapz(newt,(beta_hat2-beta0/factor2).^2);
    else
      mise_vec1(rep,mi)=trapz(newt,(-beta_hat1-beta0/factor1).^2);
      mise_vec1_2(rep,mi)=trapz(newt,(-beta_hat2-beta0/factor2).^2);
    end
    %
    inn2=trapz(newt,est1.*beta0/factor1);
    if inn2>=0
      mise_vec2(rep,mi)=trapz(newt,(est1-beta0/factor1).^2);
      mise_vec2_2(rep,mi)=trapz(newt,(est2-beta0/factor2).^2);
    else
      mise_vec2(rep,mi)=trapz(newt,(-est1-beta0/factor1).^2);
      mise_vec2_2(rep,mi)=trapz(newt,(-est2-beta0/factor2).^2);
    end
    %
    inn3=trapz(newt,phi_cs1(:,1)'.*beta0/factor1);
    if inn3>=0
      mise_vec3(rep,mi)=trapz(newt,(phi_cs1(:,1)'-beta0/factor1).^2);
      mise_vec3_2(rep,mi)=trapz(newt,(phi_cs2(:,1)'-beta0/factor2).^2);
    else
      mise_vec3(rep,mi)=trapz(newt,(-phi_cs1(:,1)'-beta0/factor1).^2);
      mise_vec3_2(rep,mi)=trapz(newt,(-phi_cs2(:,1)'-beta0/factor2).^2);
    end
end
end

[min1,ind1]=min(mean(mise_vec1));
[min2,ind2]=min(mean(mise_vec2));
[min3,ind3]=min(mean(mise_vec3));
sd1=std(mise_vec1(:,ind1));
sd2=std(mise_vec2(:,ind2));
sd3=std(mise_vec3(:,ind3));


[min1_2,ind1_2]=min(mean(mise_vec1_2));
[min2_2,ind2_2]=min(mean(mise_vec2_2));
[min3_2,ind3_2]=min(mean(mise_vec3_2));
sd1_2=std(mise_vec1_2(:,ind1_2));
sd2_2=std(mise_vec2_2(:,ind2_2));
sd3_2=std(mise_vec3_2(:,ind3_2));

if err_ind==1
mise_chi1(:,2*ap_n-1)=[min1,min2,min3]';
mise_chi1(:,2*ap_n)=[sd1,sd2,sd3]';
mise_chi2(:,2*ap_n-1)=[min1_2,min2_2,min3_2]';
mise_chi2(:,2*ap_n)=[sd1_2,sd2_2,sd3_2]';
elseif err_ind==2
mise_pois1(:,2*ap_n-1)=[min1,min2,min3]';
mise_pois1(:,2*ap_n)=[sd1,sd2,sd3]';
mise_pois2(:,2*ap_n-1)=[min1_2,min2_2,min3_2]';
mise_pois2(:,2*ap_n)=[sd1_2,sd2_2,sd3_2]';
end
end
end

save('s_m1_chi1_n100.txt','mise_chi1','-ascii')
save('s_m1_chi2_n100.txt','mise_chi2','-ascii')
save('s_m1_pois1_n100.txt','mise_pois1','-ascii')
save('s_m1_pois2_n100.txt','mise_pois2','-ascii')