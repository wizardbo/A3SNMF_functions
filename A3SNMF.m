function [C,zero_idxs] = A3SNMF(Y,new_S,Z,D,V,C,alpha,lambda1,lambda2,n,l)   

    mu = 1;
    Theta1 = sparse(n,n);
    Theta2 = sparse(n,l);
    outDim = l;

    tol=1e-3;
    maxIter = 5;
    display_iter = 1;
    tic;
    t1 = clock;    
    for iter = 1:maxIter
        [V]             = A3SNMF_Update_V(new_S,Z,D,V,alpha);        
        [C, zero_idxs]  = A3SNMF_Update_C(D,Theta2,lambda2,mu); 
        [D]             = A3SNMF_Update_D(new_S,Z,D,V,C,Theta2,mu,alpha);
%         [C, nnz_idxs, zero_idxs]  = new_A3SNMF_Update_C(D,Theta2,lambda2,mu);   
        [G]             = A3SNMF_Update_G(new_S,Z,Theta1,mu);
        [Z]             = A3SNMF_Update_Z(new_S,G,Theta1,C,V,lambda1,mu);
        
%         [~,prec_label] = maxk(C,1,2); 
%         tempY = Y(1,nnz_idxs);
%         tempPrec = prec_label(nnz_idxs);
%         results = get_all_metrics(Y,prec_label');     
%         fprintf("iter %d -- all results -- Acc:%.2f, Nmi:%.2f, Ari:%.2f, zero num: %d\n", iter, results.acc, results.nmi, results.ari, length(zero_idxs));   
        
        Theta1 = Theta1 + mu.*(new_S - Z - G);
        Theta2 = Theta2 + mu.*(C - D);
        
        obj1 = norm((new_S - Z - D*V'),'fro')^2;    
        obj2 = lambda1 * sum(sum(Z));      
        obj3 = lambda2 * sum(sqrt(sum(C.^2,2)));
        obj4 = alpha * norm(D-V, 'fro')^2;
        obj5 = 0.5 * norm(new_S - Z - G + Theta1, 'fro')^2;
        obj6 = 0.5 * norm(C - D + Theta2, 'fro')^2;
        obj(iter) = obj1 + obj2 + obj3 + obj4 + obj5 + obj6;        
    
        if (iter==1 || mod(iter, display_iter)==0)  
            fprintf('Iter %d, Obj: %g, Time:%g\n', iter, obj(iter), etime(clock,t1));
        end         
        
        Iters(iter) = iter;
        
        if (iter>2 && ((norm((obj(iter)-obj(iter-1)),'fro'))^2/(norm(obj(iter-1),'fro'))^2)<tol)
            break;
        end
    end
    
%     plot(Iters, obj);
end