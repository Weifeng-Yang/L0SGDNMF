
% function Lapk=LLaplace(ngmar)
% Lapk={};
%     for i=1:length(size(ngmar))
%         ngtemp=tenmat(ngmar,i);
%         ngtemp=double(ngtemp');
%         T=1:size(ngtemp,2);
%         sizea=size(ngtemp,2);
%         T=sort(repmat(T',sizea,1));
%         for j=1:sizea
%             Ta=T((j-1)*sizea+1:j*sizea);
%             ngsum1=ngtemp(:,Ta)-ngtemp;
%             ma(j,:)=double(exp(-vecnorm(ngsum1,2).^2));
%         end
%         Lapk{i}=ma;
%         clear ngs ngsum1 ngtemp ma
%     end
% 
% end

function [Lapk,dma,ma]=LLaplace(ngmar)
Lapk={};
    for i=1:length(size(ngmar))
        Lapk{i}=zeros(size(ngmar,i),size(ngmar,i));
    end

        i=length(size(ngmar));
        ngtemp=tenmat(ngmar,i);
        ngtemp=double(ngtemp');
        T=1:size(ngtemp,2);
        sizea=size(ngtemp,2);
        T=sort(repmat(T',sizea,1));
        for j=1:sizea
            Ta=T((j-1)*sizea+1:j*sizea);
            ngsum1=ngtemp(:,Ta)-ngtemp;
            ma(j,:)=double(exp(-vecnorm(ngsum1,2).^2));
%             ma1(:,j)=double(exp(-vecnorm(ngsum1,2).^2));
        end
        %%避免自环所带来的数值下溢，来计算dma
        ma=ma-eye(size(ma));
        dma=diag(sum(ma));
        Lapk{i}=dma-ma;
        clear ngs ngsum1 ngtemp 
end






