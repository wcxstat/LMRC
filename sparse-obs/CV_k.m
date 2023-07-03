%% K-fold CV function
function value=CV_k(t,u,y,h,K)
n=length(t);
M=floor(n/K);
ycell=cell(1,n);
for i=1:n
    N=length(t{i});
    ycell{i}=repmat(y(i),[1,N]);
end
%----calculate the cross validation criterion function------
value=0;
for k=1:K
    if k<K
        index=(1+(k-1)*M):(k*M);
    else
        index=(1+(K-1)*M):n;
    end
    index1=setdiff(1:n,index);
    t_k=t(index1);
    u_k=u(index1);
    ycell_k=ycell(index1);
    Tvec_k=cell2mat(t_k);
    uvec_k=cell2mat(u_k);
    Yvec_k=cell2mat(ycell_k);
    xin_k=[Tvec_k;Yvec_k];
    yin_k=uvec_k';
    win_k=ones(1,length(Tvec_k));
    cv=0;
    for j=1:length(index)
        out1=t{index(j)};
        out2=y(j);
        [invalid,EXY]=...
            mullwlsk_new(h,'gauss',xin_k,yin_k,win_k,out1,out2(1));
        if invalid==1
            value=1e+308;
            break;
        end
        cv=cv+sum((EXY-u{index(j)}).^2);
    end
    if invalid==1
        break;
    end
    value=value+cv;
end
value=value/(length(cell2mat(t)));