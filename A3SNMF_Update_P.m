function [P] = A3SNMF_Update_P(X,C)
    P = (C'*X)./sum(C,1)';
end

