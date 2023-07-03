function [lambda,phi1,phi2]=FCS_sparse(kerfun,xcov,kappa,eigfun,arg,noeig)

ngrid=length(arg); 
phi_x=eigfun(:,1:noeig);
xcov_inv=phi_x*diag(kappa(1:noeig).^(-1))*phi_x';

h=range(arg)/(ngrid-1);
xcov_inv2=xcov_inv;
xcov_inv2(:,[1,end])=xcov_inv2(:,[1,end])/2;
Gamma=xcov_inv2*kerfun*h;

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