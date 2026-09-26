%% Underflow handling of lipchitz constants
function ck=checkck(ck)
    if(ck<10^-8) 
       ck=10^-8;
     end
end
