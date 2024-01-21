function [Z] = A3SNMF_Update_Z(S,G,theta,D,V,lambda2,mu1)
tic;
t1 = clock;
    tempS = (2+mu1).*S - mu1.*G + theta;
    [i,j,v]      = find(tempS);    
    DV = D(i,:).*V(j,:);        
    new_v = v - 2.*sum(DV,2);  
    
    Z = sparse(i,j,new_v)./(2 + mu1 + 2*lambda2);
    Z = max(Z,0);

%     new_v = Softthres(new_v,lambda2/2);    
%     Z = sparse(i,j,new_v);
    
% fprintf("Update Z Cost:%g \n",etime(clock,t1));
end

