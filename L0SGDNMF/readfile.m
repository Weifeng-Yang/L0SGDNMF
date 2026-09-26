function [ngmar,R,tlabel]=readfile(i)
  if(i==1) 
   E=load('.\Data\FashionMnist.mat');
   ngmar= double(E.fea);
   label= double(E.gnd);


    M11=double(ngmar);
    ngmar=normalize(M11,'range');


    
   tlabel=double(label);
    R=length(unique(tlabel));
     if(find(tlabel==0)~=0)
            tlabel=tlabel(:,1)+1;
     end
    rng(2027+i);
    len=100;
    labelu=unique(tlabel);  
    label=[];
    id=[];
     for j=1:R
         temp=find(tlabel==labelu(j));
         temp=temp(randperm(length(temp)));
         id=[id;temp(1:len)];
         label=[label,ones(1,len)*j];
     end
     tlabel=tlabel(id);
     ngmart=ngmar(:,id);
     ngmar=ngmart;





   elseif(i==2)    
   E=load('.\Data\pixraw10P.mat');
   ngmar= double(E.X);
   label= double(E.Y);


    M11=double(ngmar)';
    ngmar=normalize(M11,'range');



   tlabel=double(label);
    R=length(unique(tlabel));

  end



end



