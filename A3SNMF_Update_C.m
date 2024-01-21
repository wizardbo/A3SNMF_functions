function [C, zero_idxs] = A3SNMF_Update_C(D,Theta2,lambda3,mu2)
tic;
t1 = clock;

    B = D - (Theta2)./mu2;
%     B = max(B,0);

    [n,l] = size(D); 
    C = sparse(n,l);
    
%     ids = find(sum(B,2) > 0);
%     B(ids,:) = B(ids,:)./sum(B(ids,:),2);
    
    B_row_norm = sqrt(sum(B.^2,2));
    nnz_idxs  = find(B_row_norm > (lambda3/mu2));
    
%     C(nnz_idxs,:) = max(B(nnz_idxs,:) .* (1 - (lambda3/mu2)./B_row_norm(nnz_idxs)),0);
    C(nnz_idxs,:) = B(nnz_idxs,:) .* (1 - (lambda3/mu2)./B_row_norm(nnz_idxs));
    
    zero_idxs = setdiff(1:n, nnz_idxs);    
    C(zero_idxs,:) = 0;
    
% fprintf("Update C Cost:%g\n,",etime(clock,t1));
end

