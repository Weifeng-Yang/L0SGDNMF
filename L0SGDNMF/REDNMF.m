%%  All parameters of this function are explained the same as 'main_Run_me' and 'ALGOchoose' functions
function [var,loss,timerun]=REDNMF(var,ngmar,maxiteropt,stopindex,beta)
%% initialization algorithm
loss=[];
lamda=2;
alphat=0.1;
timerun=[0];
num=length(size(ngmar));
R=size(var{1},2);
var=[];

for i=1:num
    var{i}=rand(size(ngmar,i),R);
end
var{num}=var{num}';


[~,dma,ma]=LLaplace(ngmar);
LK=zeros(1,num);
L=ones(1,num);
varK=var;
returnloss=norm(ngmar,"fro");


for i=1:num
    alpha(i)=0;
end
alpha(num)=alphat;




t1=clock;


for i=1:maxiteropt
%% update parameters
fprintf("%d\n",i);
varK=var;
for j=1:num
    LK(j)=L(j);
    [V,L(j)]=grad(var,ngmar,j,lamda,alpha,beta,dma,ma);
    var{j}=V;    
end
% loss(i+1)=compute(var,num,ngmar)+alpha(num)*norm(var{num},'fro')^2;





%% Check if termination condition is met
fprintf("REDNMF\n");
check1=0;
check2=0;
for j=1:num
    check1=check1+norm(var{j}-varK{j},'fro');
    check2=check2+norm(varK{j},'fro');
end
t2=clock;
timerun(i+1)=etime(t2,t1);
Res=check1/check2;
fprintf("Rel：%d\n",Res);
stop=stopcheck(Res,timerun,stopindex);
if(stop==1)
    fprintf("Number of terminations：%d\n",i);
    pause(2);
    break;
end




end

end














function loss=compute(var,num,ngmar)
    nga=var{1};
    for i=2:num
        nga=nga*var{i};
    end
    loss=norm(ngmar-nga,'fro');
end


function [U,L]=grad(var,ngmar,n,lamda,alpha,beta,dma,ma)
    L=1;
    alpha=alpha(end);
    if(beta~=1)
        if(n==1)
        U=ngmar.*(var{1}*var{2}).^(beta-2)*var{2}'+(lamda*var{2}.*(var{1}'*ngmar).^(beta-2)*ngmar')';
        U1=(var{1}*var{2}).^(beta-1)*var{2}'+(lamda*(var{1}'*ngmar).^(beta-1)*ngmar')';
        U=var{1}.*(U./U1);
        else
        U=var{1}'*(ngmar.*(var{1}*var{2}).^(beta-2))+(lamda/(beta-1))*(var{1}'*ngmar).^(beta-2)+alpha*var{2}*ma;
        U1=var{1}'*(var{1}*var{2}).^(beta-1)+(lamda/(beta-1))*var{2}.^(beta-1)+alpha*var{2}*dma;
        U=var{2}.*(U./U1);
        end
    else
        if(n==1)
        U=ngmar.*(var{1}*var{2}).^(beta-2)*var{2}'+(lamda*var{2}.*(var{1}'*ngmar).^(beta-2)*ngmar')';
        U1=(var{1}*var{2}).^(beta-1)*var{2}'+(lamda*(var{1}'*ngmar).^(beta-1)*ngmar')';
        U=var{1}.*(U./U1);
        else
        U=var{1}'*(ngmar./(var{1}*var{2}))+lamda*log(var{1}'*ngmar)+alpha*var{2}*ma;
        U1=var{1}'*eye(size(var{1},1),size(var{2},2))+lamda*log(var{2})+alpha*var{2}*dma;
        U=var{2}.*(U./U1);

        end
    end
    
end




