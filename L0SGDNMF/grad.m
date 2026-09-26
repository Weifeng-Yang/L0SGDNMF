%% Gradient calculation function for DNMF
function [U,L]=grad(var,ngmar,Lapk,n,num,r,alpha)
    if(n>1)
%     Z=eye(size(var{1},1),size(var{1},1));
    Z=var{1};
    end
    if(n<num)
%     ZT=eye(size(var{n+1},1),size(var{n+1},1)); 
    ZT=var{n+1};
    end
    
%     tic
%     for i=1:n-1
%         Z=Z*var{i};
%     end
%     for i=n+1:num
%         ZT=ZT*var{i};
%     end 
%     toc

%     tic
   
    for i=2:n-1
        Z=Z*var{i};
    end

    
    for i=n+2:num
        ZT=ZT*var{i};
    end 
%     toc


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
    L=ck+alpha(end)*norm(Lapk{end},'fro');
    L=checkck(L);
    mar=Z*var{n};
    U=var{n}-1/(r*L)*(Z'*(mar-ngmar)+alpha(n)*var{n}*Lapk{end});   
    end
end


