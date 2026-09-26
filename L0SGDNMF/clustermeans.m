function [acc,rdx,NMIs]=clustermeans(vartemp,N,y)
        len=10;

        for k=1:len
        cluster(:,k)=kmeans(vartemp, N,'Distance','cosine','Maxiter',1000); 
        NMItemp(:,k)=NMI(y,cluster(:,k));
        end
        
%         acc=0;
%         rdx=0;
%         NMIs=0;
        
        len1=size(cluster,2);

        for k=1:len1
        
        acc(k)=cluster_acc(y,cluster(:,k));
        rdx(k)=ARindex(y,cluster(:,k),'adjusted');
        NMIs(k)=NMI(y,cluster(:,k)); 
        end

        acc=sum(acc)/len1;

        rdx=sum(rdx)/len1;

        NMIs=sum(NMIs)/len1;
end
