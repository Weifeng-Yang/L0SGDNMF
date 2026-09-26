%% Deep non-negative matrix factorization with multi-layer graph regularization
% var      : Initial matrices {W_1,...,W_l,H_l}
% ngmar    : Original data matrix, whose columns are samples
% alphat   : Scalar shared by all graph terms, or one parameter for each layer
%
% The function follows the input and output form of the existing baselines.
% The returned var is {W_1,...,W_l,H_l}, so var{end}' can be directly used
% by the existing clustering code.

function [var,loss,timerun]=MGDNMF(var,ngmar,maxiteropt,stopindex,alphat)

num=length(var);
layernum=num-1;
eps0=1e-12;
pretrainiter=100;
loss=zeros(1,maxiteropt+1);
timerun=zeros(1,maxiteropt+1);

if(layernum<1)
    error('MGDNMF requires at least one basis matrix and one representation matrix.');
end

if(isscalar(alphat))
    alpha=ones(1,layernum)*alphat;
elseif(numel(alphat)==layernum)
    alpha=reshape(alphat,1,layernum);
else
    error('alphat must be a scalar or contain one value for each layer.');
end

if(any(alpha<0))
    error('The graph regularization parameters must be nonnegative.');
end

ngmar=max(double(ngmar),0);
W=cell(1,layernum);
H=cell(1,layernum);

%% Layer-wise NMF initialization
% The intermediate representations H_1,...,H_{l-1} are required by
% MGDNMF but are not stored in the original experimental framework.
% They are initialized from the supplied matrix chain and then pretrained
% layer by layer without changing the external interface.
Hprevious=ngmar;
for j=1:layernum
    W{j}=max(double(var{j}),eps0);

    H{j}=double(var{j+1});
    for q=j+2:num
        H{j}=H{j}*double(var{q});
    end
    H{j}=max(H{j},eps0);

    if(size(Hprevious,1)~=size(W{j},1) || size(W{j},2)~=size(H{j},1) ...
            || size(Hprevious,2)~=size(H{j},2))
        error('The dimensions of the matrices in var are incompatible with ngmar.');
    end

    for i=1:pretrainiter
        Wnumerator=Hprevious*H{j}';
        Wdenominator=W{j}*(H{j}*H{j}');
        W{j}=max(W{j}.*(Wnumerator./max(Wdenominator,eps0)),eps0);

        Hnumerator=W{j}'*Hprevious;
        Hdenominator=(W{j}'*W{j})*H{j};
        H{j}=max(H{j}.*(Hnumerator./max(Hdenominator,eps0)),eps0);
    end
    Hprevious=H{j};
end

%% Construct the graph of the original data and each intermediate layer
% LLaplace.m is reused without changing its graph construction rule.
Lapk=cell(1,layernum);
dma=cell(1,layernum);
ma=cell(1,layernum);
for j=1:layernum
    if(j==1)
        graphdata=ngmar;
    else
        graphdata=H{j-1};
    end
    [Ltemp,dma{j},ma{j}]=LLaplace(graphdata);
    Lapk{j}=Ltemp{end};
end

dmasum=zeros(size(dma{1}));
masum=zeros(size(ma{1}));
for j=1:layernum
    dmasum=dmasum+alpha(j)*dma{j};
    masum=masum+alpha(j)*ma{j};
end

loss(1)=compute(W,H,ngmar,Lapk,alpha);
t1=clock;
lastiter=maxiteropt;

for i=1:maxiteropt
    fprintf("%d\n",i);
    WK=W;
    HK=H;

    %% Update W_1,...,W_l
    for j=1:layernum
        if(j==1)
            Hprevious=ngmar;
        else
            Hprevious=H{j-1};
        end

        Wnumerator=Hprevious*H{j}';
        Wdenominator=W{j}*(H{j}*H{j}');
        W{j}=max(W{j}.*(Wnumerator./max(Wdenominator,eps0)),eps0);
    end

    %% Update H_1,...,H_{l-1}
    for j=1:layernum-1
        if(j==1)
            Hprevious=ngmar;
        else
            Hprevious=H{j-1};
        end

        Hnumerator=W{j}'*Hprevious+W{j+1}*H{j+1};
        Hdenominator=(W{j}'*W{j})*H{j}+H{j};
        H{j}=max(H{j}.*(Hnumerator./max(Hdenominator,eps0)),eps0);
    end

    %% Update the final low-dimensional representation H_l
    if(layernum==1)
        Hprevious=ngmar;
    else
        Hprevious=H{layernum-1};
    end
    Hnumerator=W{layernum}'*Hprevious+H{layernum}*masum;
    Hdenominator=(W{layernum}'*W{layernum})*H{layernum}+H{layernum}*dmasum;
    H{layernum}=max(H{layernum}.*(Hnumerator./max(Hdenominator,eps0)),eps0);

    loss(i+1)=compute(W,H,ngmar,Lapk,alpha);
    t2=clock;
    timerun(i+1)=etime(t2,t1);

    fprintf("MGDNMF\n");
    check1=0;
    check2=0;
    for j=1:layernum
        fprintf("nonzero:%d\n",nnz(W{j}));
        check1=check1+norm(W{j}-WK{j},'fro');
        check2=check2+norm(WK{j},'fro');
    end
    for j=1:layernum
        check1=check1+norm(H{j}-HK{j},'fro');
        check2=check2+norm(HK{j},'fro');
    end
    fprintf("nonzero:%d\n",nnz(H{layernum}));

    Res=check1/max(check2,eps0);
    fprintf("Rel:%d\n",Res);
    stop=stopcheck(Res,timerun,stopindex);
    if(stop==1)
        fprintf("Number of terminations:%d\n",i);
        lastiter=i;
        break;
    end
end

loss=loss(1:lastiter+1);
timerun=timerun(1:lastiter+1);

for j=1:layernum
    var{j}=W{j};
end
var{end}=H{layernum};

end


function loss=compute(W,H,ngmar,Lapk,alpha)
layernum=length(W);
loss=0;

for j=1:layernum
    if(j==1)
        Hprevious=ngmar;
    else
        Hprevious=H{j-1};
    end
    loss=loss+norm(Hprevious-W{j}*H{j},'fro')^2;
    loss=loss+alpha(j)*trace(H{layernum}*Lapk{j}*H{layernum}');
end

loss=real(loss);
end
