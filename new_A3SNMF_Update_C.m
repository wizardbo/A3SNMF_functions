function [C, nnz_idxs, zero_idxs] = new_A3SNMF_Update_C(D,Theta2,lambda2,mu2)
tic;
t1 = clock;

    B = max(D - Theta2./mu2, 0);    

    [n,l] = size(D); 
    C = sparse(n,l);
    
    sum_B = sum(B,2);
    B = B./sum_B;
    
    B_row_norm = sqrt(sum(B.^2,2));
    nnz_idxs  = find(B_row_norm > lambda2/mu2);
    zero_idxs = setdiff(1:n, nnz_idxs);
    
    
%     C(nnz_idxs,:) = sum_B(nnz_idxs,:) .* B(nnz_idxs,:) .* (1 - (lambda2/mu2)./B_row_norm(nnz_idxs));
    C(nnz_idxs,:) = sum_B(nnz_idxs,:) .* B(nnz_idxs,:) .* (1 - (lambda2/mu2)./B_row_norm(nnz_idxs));
    C(zero_idxs,:) = 0;
    
    
    
% fprintf("UpdateU Cost:%g,",etime(clock,t1));
end

