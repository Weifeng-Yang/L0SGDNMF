%%  All parameters of this function are explained the same as 'main_Run_me' and 'ALGOchoose' functions
function [var,loss,timerun]=L0SGDNMF(var,ngmar,aa,maxiteropt,stopindex,r,btmax,rho,alphat)
%% initialization algorithm

loss=[];

timerun=[0];


num=length(var);
for i=1:num
    varze{i}=zeros(size(var{i}));
end
Lapk=LLaplace(ngmar);




LK=zeros(1,num);
L=ones(1,num);
tk=1;
bts=[];
wk=zeros(1,num);
varK=var;


returnloss=norm(ngmar,"fro")^2;

for i=1:num-1
    alpha(i)=0;
end
alpha(num)=alphat;
loss(1)=compute(var,num,ngmar)+alpha(num)/2*trace(var{num}*Lapk{end}*var{num}');





t1=clock;


for i=1:maxiteropt
%% update parameters
fprintf("%d\n",i);
vv=var;
tnew=(1+sqrt(1+4*tk^2))/2;

%% Update parameters
for j=1:num
    wk(j)=min((tk-1)/tnew,btmax);
    vv{j}=var{j}+wk(j)*(var{j}-varK{j});
    varY{j}=vv{j};
    varK{j}=var{j};
    LK(j)=L(j);
    [V,L(j)]=grad(vv,ngmar,Lapk,j,num,r,alpha);
    if(j<num)
        var{j}=PROX(varze{j},V,aa(j));
    else
        var{j}=PROXn1(V);
    end
    vv{j}=var{j};
end
loss(i+1)=compute(var,num,ngmar)+alpha(num)/2*trace(var{num}*Lapk{end}*var{num}');

sumnorm=0;
for j=1:length(var)
    sumnorm=sumnorm+norm(var{j}-varY{j},'fro')^2;
end

%% Judging whether to extrapolate
if(loss(i+1)>loss(i)-rho*sumnorm)
    var=varK;
    for j=1:num
    [V,L(j)]=grad(var,ngmar,Lapk,j,num,r,alpha);
    if(j<num)
        var{j}=PROX(varze{j},V,aa(j));
    else
        var{j}=PROXn1(V);
    end
    end
    loss(i+1)=compute(var,num,ngmar)+alpha(num)/2*trace(var{num}*Lapk{end}*var{num}');
end


bts{i}=wk;
t2=clock;
timerun(i+1)=etime(t2,t1);
if(alphat==0)
fprintf("L0SDNMF\n");
else
fprintf("L0SGDNMF\n");
end
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
fprintf("cri:%d\n",Res);
stop=stopcheck(Res,timerun,stopindex);
if(stop==1)
    fprintf("Number of terminations:%d\n",i);
    pause(4);
    break;
end


tk=tnew;

end

end


function x=PROXn1(x)
x=double(x);
x(x<0)=0;
end



