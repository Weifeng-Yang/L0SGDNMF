clearvars -except 
clc
warning('off');

%% Parameter.
%   dimension : The ranks of matrix decomposition
%   index     : The dataset to be used, when index=1, use FashionMNIST dataset
%               when index=2, use Pixraw10P dataset. 
%   r         : Step factor
%   maxiteropt: Maximum iteration alloted to the method
%   trigger   : Whether to enable the indicator array of each method, where
%               when 1-trigger, enable the l0-NMF method
%               when 2-trigger, enable the GSNMF method
%               when 3-trigger, enable the l1-SDNMF method
%               when 4-trigger, enable the S-GNMFSC method
%               when 5-trigger, enable the GDNMF method
%               when 6-trigger, enable the DHGLpSNMF method
%               when 7-trigger, enable the l0-SDNMF method
%               when 8-trigger, enable the MGDNMF method
%               when 9-trigger, enable the l1-SGDNMF method
%               when 10-trigger, enable the l0-SGDNMF method 
%   percent   : The proportion of non-zero elements allowed in each decomposition matrix
%   alphat    : The graph regularization parameter
%   stopindex : The indicator of the stop condition.  
%               To set the specific termination condition, see the 'stopcheck' function for details.  
%               The default termination condition is: Rel<1e-5 or maxiteropt>8000 
%% Display
%   nonzero   : The number of non-zero elements in each component.
%   Rel       : The difference in the variable value between two iterations.


%% Parameter settings and Select dataset
rng('shuffle')
index=1;
maxiteropt=8000;
r=1.01;
rho=10^-8;
btmax=0.9999;
alphat=1;
outer=10;
trigger=[10];
percent=[0.7,0.7,1];
stopindex=4;
[ngmar,N,y]=readfile(index);
dimension=[size(ngmar,1),100,ceil(size(ngmar,2)/4),size(ngmar,2)];




    
num=length(dimension)-1;
for i=1:num
    aa(i)=ceil(dimension(i)*dimension(i+1)*percent(i));
end


for j=1:outer
%% Init
for i=1:num
    var{i}=rand(dimension(i),dimension(i+1));
end


%% Solving
for i=1:length(trigger)       
[datas{i},vars{i}]=ALGOchoose(var,ngmar,aa,maxiteropt,trigger(i),stopindex,r,btmax,rho,alphat);
end
datas{length(trigger)+1}=vars;
datass{j}=datas;

for i=1:length(trigger)
    vars1=datas{length(trigger)+1};
    vartemp=vars1{i};
    [acc(j,i),rdx(j,i),NMIs(j,i)]=clustermeans(vartemp{end}',N,y);
end


end




 
 
%% Display results
accmean=mean(acc)
rdxmean=mean(rdx)
nmimean=mean(NMIs)
plt0=plotplt(trigger,acc,rdx,NMIs); 
 
 
 
 




