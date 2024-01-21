function [centers] = CFC(X,prec_label,l,d)
    v = size(X,2);
    is_cluster_contain_word = zeros(l,v);
    centers = zeros(l,v);
    for i = 1:l
       idxs = find(prec_label==i);
       if isempty(idxs)
           centers(i,:) = 1;
       else
           is_cluster_contain_word(i,:) = any(X(idxs,:));
           centers(i,:) = d.^(sum(X(idxs,:))/(length(idxs)));
       end
    end
    
    sum_cluster_contain_word = sum(is_cluster_contain_word);
    sum_cluster_contain_word(find(sum_cluster_contain_word == 0)) = l;
    
    centers = centers .* log(l./sum_cluster_contain_word);
    centers(isinf(centers)) = 0;
end

