%addpath(genpath('E:\xuwenchao\MAFLQR\PACE_matlab'));
addpath(genpath('/Users/xuwenchao/Documents/PACE_matlab'));
newt=linspace(0,1,101);
opt=6;
p=setOptions('regular',0,'error',1,'selection_k',opt,'kernel','gauss',...
    'newdata',newt);

mvec=1:4;
n=300;
Jmax=8;
L=length(newt);
PHI=[ones(1,L);sqrt(2)*cos((1:(Jmax-1))'*newt*pi)]; % Jamx-by-L matrix
beta_coef=4*(-1).^(2:(Jmax+1)).*(1:Jmax).^(-3);
beta0=beta_coef*PHI; % true slope function

alphavec=[1.1,1.5,1.8];
mise_chi=zeros(2,6);
mise_pois=zeros(2,6);
for err_ind=1:2
for ap_n=1:length(alphavec)
alpha=alphavec(ap_n);
factor1=5*sqrt(sum((1:Jmax).^(-alpha).*(beta_coef.^2)));
factor2=sqrt(sum(beta_coef.^2));

mise_vec=zeros(100,length(mvec));
mise_vec_2=zeros(100,length(mvec));
for rep=1:100
fprintf('error=%d, alpha=%d, and repeat=%d\n',err_ind,alpha,rep)
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
xcov=eigfun*diag(kappa)*eigfun';

%%%%%%%% IRLD %%%%%%%%
Tvec=cell2mat(t);
uvec=cell2mat(u);
Yvec=ones(1,length(Tvec));
ia=0;
for i=1:n
    index=(ia+1):(ia+N(i));
    Yvec(index)=repmat(Y(i),1,N(i));
    ia=ia+N(i);
end
xin=[Tvec;Yvec];
yin=uvec';
win=ones(1,length(Tvec));
out1=newt;
out2=sort(Y);

[bw1,~]=gcv_lwls(yin',Tvec,'epan',-2,1,0,2,'off',0);
bw2=max(out2(2:end)-out2(1:(end-1)));
h0=[bw1,bw2];
%bw1_cand=max(bw1-0.5,0.1):0.1:(bw1+0.5);
% bw1_cand=0.1:0.1:1;
% bw2_cand=max(bw2-5,5):0.1:(bw2+5);
% value=zeros(length(bw1_cand),length(bw2_cand));
% for i=1:length(bw1_cand)
%     for j=1:length(bw2_cand)
%         h=[bw1_cand(i),bw2_cand(j)];
%         value(i,j)=CV_k(t,u,Y,h,5);
%     end
% end
% [val1,ind1]=min(value);
% [~,ind2]=min(val1);
% h_cv=[bw1_cand(ind1(ind2)),bw2_cand(ind2)];

options=optimset('MaxFunEvals', 1000, 'MaxIter', 1000,'TolX',1e-10,'TolFun',1e-10, 'Display', 'off' );
h_cv=fmincon(@(h)CV_k(t,u,Y,h,5), h0, [],[], [],[], [0,max(bw2-5,4)], [1,bw2+5],[], options);

[invalid,EXY]=mullwlsk_new(h_cv,'gauss',xin,yin,win,out1,out2);
Gamma_e=(EXY-mean(EXY))'*(EXY-mean(EXY))/n;
%%%%%%%%%%%%%%%%

for mi=1:length(mvec)
m=mvec(mi);
[~,phi_irld1,phi_irld2]=IRLD(Gamma_e,xcov,kappa,eigfun,newt,m);
inn=trapz(newt,phi_irld1(:,1)'.*beta0/factor1);
if inn>=0
  mise_vec(rep,mi)=trapz(newt,(phi_irld1(:,1)'-beta0/factor1).^2);
  mise_vec_2(rep,mi)=trapz(newt,(phi_irld2(:,1)'-beta0/factor2).^2);
else
  mise_vec(rep,mi)=trapz(newt,(-phi_irld1(:,1)'-beta0/factor1).^2);
  mise_vec_2(rep,mi)=trapz(newt,(-phi_irld2(:,1)'-beta0/factor2).^2);
end
end
end

[min1,ind1]=min(mean(mise_vec));
sd1=std(mise_vec(:,ind1));

[min1_2,ind1_2]=min(mean(mise_vec_2));
sd1_2=std(mise_vec_2(:,ind1_2));

if err_ind==1
mise_chi(:,2*ap_n-1)=[min1,min1_2]';
mise_chi(:,2*ap_n)=[sd1,sd1_2]';
elseif err_ind==2
mise_pois(:,2*ap_n-1)=[min1,min1_2]';
mise_pois(:,2*ap_n)=[sd1,sd1_2]';
end
end
end

round(mise_chi*100,3)
round(mise_pois*100,3)