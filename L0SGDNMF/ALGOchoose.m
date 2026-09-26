%% Input.
% var         : Initial matrix
% ngmar       : Decomposed matrix
% aa          : Maximum number of non-zero elements for each decomposition matrix
% The remaining parameters are explained the same as the 'main_Run_me' function

%% Output.
% vars:       : Decomposition matrix resulting from the final iterative result
% loss:       : Array of loss functions generated during iteration
% tr:         : Runtime array during iteration
% btss and atss: An array of extrapolation parameters produced by each algorithm during iteration,


%% Select method

function [data,varss]=ALGOchoose(var,ngmar,aa,maxiteropt,flag,stopindex,r,btmax,rho,alphat)

if(flag==1) 
[vars,loss,tr]=L0NMF(var,ngmar,aa,maxiteropt,stopindex,r);
varss=vars;
lossdata=loss;
trdata=tr;


elseif(flag==2)
[vars,loss,tr]=L1GSNMF(var,ngmar,maxiteropt,stopindex,r,1);
varss=vars;
lossdata=loss;
trdata=tr;


elseif(flag==3) 
[vars,loss,tr]=SDNMF(var,ngmar,maxiteropt,stopindex,r,btmax,rho,1);
varss=vars;
lossdata=loss;
trdata=tr;

elseif(flag==4) 
[vars,loss,tr]=GNMFOS(var,ngmar,maxiteropt,stopindex,r,alphat);
varss=vars;
lossdata=loss;
trdata=tr;

elseif(flag==5) 
[vars,loss,tr]=REDNMF(var,ngmar,maxiteropt,stopindex,2);
varss=vars;
lossdata=loss;
trdata=tr;

elseif(flag==6) 
[vars,loss,tr]=DHGLpSNMF(var,ngmar,maxiteropt,stopindex,r,0.005,0.8);
varss=vars;
lossdata=loss;
trdata=tr;





elseif(flag==7) 
[vars,loss,tr]=L0SGDNMF(var,ngmar,aa,maxiteropt,stopindex,r,btmax,rho,0);  
varss=vars;
lossdata=loss;
trdata=tr;

elseif(flag==8) 
[vars,loss,tr]=MGDNMF(var,ngmar,maxiteropt,stopindex,alphat);  
varss=vars;
lossdata=loss;
trdata=tr;


elseif(flag==9) 
[vars,loss,tr]=L1GDNMF(var,ngmar,maxiteropt,stopindex,r,btmax,rho,alphat,0.5);  
varss=vars;
lossdata=loss;
trdata=tr;

elseif(flag==10) 
[vars,loss,tr]=L0SGDNMF(var,ngmar,aa,maxiteropt,stopindex,r,btmax,rho,alphat);  
varss=vars;
lossdata=loss;
trdata=tr;






end











data{1}=lossdata;
data{2}=trdata;


end



