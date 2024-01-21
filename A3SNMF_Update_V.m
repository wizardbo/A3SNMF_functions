function [V] = A3SNMF_Update_V(S,Z,D,V,alpha)
tic;
t1 = clock;

    [~,k] = size(D);
    
    A = (S - Z);    
%     clear S Z;
    A = [A; sqrt(alpha).*D'];    
    D = [D; sqrt(alpha).*eye(k)];

    Z1 = A'*D;      clear A;
    UU = D'*D;
    
    diag_UU = diag(UU);
    
    for i = 1:k
        Z2 = V*UU;
        Z = Z1 - Z2;
        V(:,i) = max(V(:,i) + Z(:,i)./diag_UU(i), 0);
    end
    
    clear UU Z1 Z2;
end

