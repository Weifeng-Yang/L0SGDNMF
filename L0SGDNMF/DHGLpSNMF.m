%%  DHGLpSNMF
function [var,loss,timerun]=DHGLpSNMF(var,ngmar,maxiteropt,stopindex,~,beta,p)



ngmar=double(ngmar);
num=length(var);



layers=zeros(1,num-1);

for i=1:num-1
    layers(i)=size(var{i},2);
end

lambda=0.001;
neighbors=5;
t1=tic;
Z=cell(1,num-1);
H=cell(1,num-1);
L=cell(1,num-1);

%% Layer-wise pretraining released with DHGLpSNMF
for i=1:num-1
    if(i==1)
        V=ngmar;
    else
        V=H{i-1};
    end

    L{i}=HypergraphMatrix(V,neighbors);
    smoothlambda=(i==num-1)*lambda;
    [Z{i},H{i}]=ShallowPretraining(V,layers(i),maxiteropt,L{i},smoothlambda,p);
end

Herror=DeepRepresentations(Z,H);
vars=[Z,{H{end}}];
loss=zeros(1,maxiteropt+1);
timerun=zeros(1,maxiteropt+1);
loss(1)=ObjectiveValue(ngmar,Z,H,L,beta,lambda,p);
timerun(1)=toc(t1);
lastiter=maxiteropt;

%% Forward-backward fine-tuning
for iter=1:maxiteropt
    varsK=vars;
    D=[];

    for i=1:num-1
        if(i==1)
            HVt=Herror{1}*ngmar';
            HHt=Herror{1}*Herror{1}';
            Z{i}=UpdateFirstBasis(Z{i},HHt,HVt);
            D=Z{1};
        else
            VHt=ngmar*Herror{i}';
            HHt=Herror{i}*Herror{i}';
            Z{i}=UpdateInnerBasis(Z{i},HHt,VHt,D);
            D=D*Z{i};
        end

        if(i==num-1)
            WtV=D'*ngmar;
            WtW=D'*D;
            H{i}=UpdateRepresentation(H{i},WtW,WtV,beta,L{i},lambda,p);
        end
    end

    Herror=DeepRepresentations(Z,H);
    vars=[Z,{H{end}}];
    loss(iter+1)=ObjectiveValue(ngmar,Z,H,L,beta,lambda,p);
    timerun(iter+1)=toc(t1);

    check1=0;
    check2=0;

    for i=1:length(vars)
        check1=check1+norm(vars{i}-varsK{i},'fro')^2;
        check2=check2+norm(varsK{i},'fro')^2;
    end

    Res=sqrt(check1)/max(sqrt(check2),eps);
    stop=stopcheck(Res,timerun,stopindex);

    if(stop==1)
        lastiter=iter;
        break;
    end
end

loss=loss(1:lastiter+1);
timerun=timerun(1:lastiter+1);
var=vars;

end


function [W,H]=ShallowPretraining(V,rank,maxiteropt,L,smoothlambda,p)

H=rand(rank,size(V,2));
W=V*pinv(H);
loss0=norm(V-W*H,'fro')^2+trace(H*L*H') ...
    +smoothlambda*sum(abs(H(:)).^p);
lossK=loss0;

for iter=1:maxiteropt
    H=UpdateRepresentation(H,W'*W,W'*V,1,L,smoothlambda,p);
    W=UpdateFirstBasis(W,H*H',H*V');
    loss=norm(V-W*H,'fro')^2+trace(H*L*H') ...
        +smoothlambda*sum(abs(H(:)).^p);

    if(iter>10)
        relative=abs(lossK-loss)/max(loss0,eps);

        if(relative<0.025)
            break;
        end
    end

    lossK=loss;
end

end


function Z=UpdateFirstBasis(Z,WtW,WtV)

Lvalue=norm(full(WtW),'fro');



gradient=Z*WtW-WtV';
Z=ProjectColumnBall(Z-gradient/Lvalue);

end


function H=UpdateInnerBasis(H,WtW,VHt,Z)

ZtZ=Z'*Z;
Lvalue=norm(full(ZtZ),'fro')*norm(full(WtW),'fro');



gradient=ZtZ*H*WtW-Z'*VHt;
H=ProjectColumnBall(H-gradient/Lvalue);

end


function H=UpdateRepresentation(H,WtW,WtV,beta,L,lambda,p)

normH=max(norm(H,'fro'),eps);
Lvalue=2*norm(full(WtW),'fro')+2*beta*norm(full(L),'fro') ...
    +lambda*p*(p-1)*normH^(p-2);



smoothgradient=lambda*p*max(abs(H),1e-12).^(p-1).*sign(H);
gradient=2*(WtW*H-WtV)+2*beta*(H*L')+smoothgradient;
H=max(H-gradient/Lvalue,0);

end


function X=ProjectColumnBall(X)

columnnorm=sqrt(sum(X.^2,1));
X=X./max(1,columnnorm);

end


function Herror=DeepRepresentations(Z,H)

num=length(H);
Herror=cell(1,num);
Herror{end}=H{end};

for i=num-1:-1:1
    Herror{i}=Z{i+1}*Herror{i+1};
end

end


function loss=ObjectiveValue(ngmar,Z,H,L,beta,lambda,p)

nga=H{end};

for i=length(Z):-1:1
    nga=Z{i}*nga;
end

loss=norm(ngmar-nga,'fro')^2;
loss=loss+beta*trace(H{end}*L{end}*H{end}');
loss=loss+lambda*sum(abs(H{end}(:)).^p);
loss=real(loss);

end


function L=HypergraphMatrix(ngmar,neighbors)

data=ngmar';
sample=size(data,1);

if(sample<=1)
    L=sparse(sample,sample);
    return;
end

neighbors=min(max(round(neighbors),1),sample-1);
sampleNorm=sum(data.^2,2);
distance=sampleNorm+sampleNorm'-2*(data*data');
distance=max(distance,0);
[~,index]=sort(distance,2,'ascend');
index=index(:,2:neighbors+1);
incidence=zeros(sample,sample);
edgeWeight=zeros(sample,1);

for i=1:sample
    incidence(i,index(i,:))=1;
    scale=max(distance(i,index(i,end)),1e-10);
    neighborDistance=distance(i,index(i,:));
    edgeWeight(i)=sum(exp(-(neighborDistance.^2)/(scale^2)));
end

W=spdiags(edgeWeight,0,sample,sample);
vertexDegree=sum(incidence*W,2);
edgeDegree=sum(incidence,1)';
vertexScale=1./sqrt(max(vertexDegree,1e-10));
edgeScale=1./max(edgeDegree,1e-10);
F=spdiags(vertexScale,0,sample,sample)*incidence*W ...
    *spdiags(edgeScale,0,sample,sample)*incidence' ...
    *spdiags(vertexScale,0,sample,sample);
L=speye(sample)-F;
L=(L+L')/2;

end
