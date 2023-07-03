function [opth,optbeta]=FSIM2(xi, y, r, initial)
% functional single index model for fixed r
% CHEN, HALL, & MULLER, (2010). Single and multiple index 
% functional regression models with nonparametric link. 
% AoS. 39, 1720–47.
% Jan 3, 2023 writen by Wenchao Xu

% Input:
% xi(n,K): FPC scores matrix (functional covariate),
%          where K is a large number
% y(n,1) : response
% r(1,1) : cut-off level
% initial(r,1): initial value for coefficients

% Output:
% opth: optimal bandwidth
% optbeta: estimated coefficients


N=length(y);
X=xi(:,1:r);
beta0=initial;

% starting values for the h's (Bowman and
% Azzalini (1997))
x_ = X*beta0;
hx=median(abs(x_-median(x_)))/0.6745*(4/3/N)^0.2;
hy=median(abs(y-median(y)))/0.6745*(4/3/N)^0.2;
h=sqrt(hy*hx);

% param starting value
param0 = [h;beta0];


%----- (2) Optimal constraints ------------------
Aeq = zeros(1, length(param0));
Aeq(2) = 1;
beq=1;

% lower boound on h
lb = [0;-Inf(length(beta0),1)];


options = optimset('Display', 'off',...
    'Algorithm', 'sqp', 'MaxFunEvals', 1e5,...
    'MaxIter', 1e5,'TolX',1e-10, 'TolFun',1e-10);
Param = fmincon(@(p) MSEg(X, y, p), param0,...
    [], [], Aeq,beq,lb, [], [] ,options);

opth = Param(1);
optbeta = Param(2:end);