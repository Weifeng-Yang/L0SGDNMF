%%  All parameters of this function are explained the same as 'main_Run_me' and 'ALGOchoose' functions
function [var,loss,timerun]=SDNMF(var,ngmar,maxiteropt,stopindex,r,btmax,rho,lamda)
%% initialization algorithm

loss=[];

timerun=[0];


num=length(var);
for i=1:num
    varze{i}=zeros(size(var{i}));
end





LK=zeros(1,num);
L=ones(1,num);
tk=1;
bts=[];
wk=zeros(1,num);
varK=var;


returnloss=norm(ngmar,"fro")^2;
loss(1)=compute(var,num,ngmar);






t1=clock;


for i=1:maxiteropt
%% update parameters
fprintf("%d\n",i);
vv=var;

%% Update parameters
for j=1:num
    wk(j)=min(wk(j),btmax);
    vv{j}=var{j}+wk(j)*(var{j}-varK{j});
    varK{j}=var{j};
    LK(j)=L(j);
    [V,L(j)]=grad(vv,ngmar,j,num,r);
    if(j<num)
        var{j}=PROXL1(V,lamda,1/(L(j)*r));
    else
        var{j}=PROXn1(V);
    end
    vv{j}=var{j};
end
loss(i+1)=compute(var,num,ngmar);

sumnorm=0;
for j=1:length(var)
    sumnorm=sumnorm+norm(var{j}-varK{j},'fro')^2;
end

%% Judging whether to extrapolate
if(loss(i+1)>loss(i)-rho/2*sumnorm)
    var=varK;
    for j=1:num
    [V,L(j)]=grad(var,ngmar,j,num,r);
    if(j<num)
        var{j}=PROXL1(V,lamda,1/(L(j)*r));
    else
        var{j}=PROXn1(V);
    end
    end
    loss(i+1)=compute(var,num,ngmar);
end


bts{i}=wk;
t2=clock;
timerun(i+1)=etime(t2,t1);
fprintf("SDNMF\n");
check1=0;
check2=0;
%% Check if termination condition is met
for j=1:num
    fprintf("nonzero:%d\n",nnz(var{j}~=0));
    check1=check1+norm(var{j}-varK{j},'fro');
    check2=check2+norm(varK{j},'fro');
end
Res=check1/check2;
% Res=abs(loss(i+1)-loss(i))/returnloss;
fprintf("cri：%d\n",Res);
stop=stopcheck(Res,timerun,stopindex);
if(stop==1)
    fprintf("Number of terminations：%d\n",i);
    pause(4);
    break;
end


tk=(1+sqrt(1+4*tk^2))/2;
for j=1:num
wk(j)=(tk-1)/(tk);
end




end



end


function x=PROXn1(x)
x=double(x);
x(x<0)=0;
end


%% Gradient calculation function for DNMF
function [U,L]=grad(var,ngmar,n,num,r)
    if(n>1)

    Z=var{1};
    end
    if(n<num)

    ZT=var{n+1};
    end
    

    for i=2:n-1
        Z=Z*var{i};
    end

    
    for i=n+2:num
        ZT=ZT*var{i};
    end 



    if(n==1)
    ck=norm(ZT*ZT','fro');
    ck=checkck(ck);
    L=ck;
    mar=var{n}*ZT;
    U=var{n}-1/(r*L)*(mar-ngmar)*ZT'; 
    elseif(n<num)
    ck=norm(Z'*Z,'fro')*norm(ZT*ZT','fro');
    ck=checkck(ck);
    L=ck;
    mar=Z*var{n}*ZT;
    U=var{n}-1/(r*L)*Z'*(mar-ngmar)*(ZT)';  
    elseif(n==num)
    ck=norm(Z'*Z,'fro');
%     ck=checkck(ck);
    L=ck;
    mar=Z*var{n};
    U=var{n}-1/(r*L)*(Z'*(mar-ngmar));   
    end
end





