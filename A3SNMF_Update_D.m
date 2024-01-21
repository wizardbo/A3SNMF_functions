function [D] = A3SNMF_Update_D(S,Z,D,V,C,Theta2,mu2,alpha)
    tic;
    t1 = clock;
    
    k = size(V,2);
%     D = ((S-Z)*V + (alpha).*V + (mu2/2).*C + Theta2./2) * (V'*V + (alpha + mu2/2).*eye(k))^-1;
%     D = max(D,0);
    
    A = (S - Z);    
%     clear S Z;
    A = [A; (sqrt(alpha+mu2/2)/(alpha+mu2/2)).*(alpha.*V + (mu2/2).*(C+Theta2./mu2))'];    
    B = [V; sqrt(alpha+mu2/2).*eye(k)];

    Z1 = A'*B;      clear A;
    UU = B'*B;
    
    diag_UU = diag(UU);
    
    for j = 1:k
        Z2 = D*UU;
        Z = Z1 - Z2;
        D(:,j) = max(D(:,j) + Z(:,j)./diag_UU(j), 0);
    end
    
    clear UU Z1 Z2;
    
%     D = D./sum(D,2);
%     D(isnan(D)) = 0;
% fprintf("UpdateV Cost:%g\n,",etime(clock,t1));    
    
%     fprintf('Update D, Time:%g\n', etime(clock,t1));
end

