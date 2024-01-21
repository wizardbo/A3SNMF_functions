function [G] = A3SNMF_Update_G(S,Z,theta,mu)
    tic;
    t1 = clock;
        
    G = S-Z+theta./mu;  
    G = max(G.*spones(S),0);
        
%     sigma = 0.1;
%     MaxIter = 10;
%     tol = 1e-5;
%     
%     for iter = 1:MaxIter  
%         % Update G
%         [U_G, Sig_G, V_G]=lansvd(full(S-Z+theta./mu),outDim, 'L');
%         Sig_G = Softthres(Sig_G,lambda1/mu);
%         
%         G = U_G*Sig_G*V_G';
%         G = max(G.*spones(S),0);       
        
%         Pdif_ratio = norm((P - Phat),'fro')/norm(P,'fro');
%         Loss_value=norm((Yhat - P*B),'fro')^2 + norm((P - X*W),'fro')^2;
% 
%         errP(iter) = lambda3 * Pdif_ratio;
%         obj(iter) = Loss_value;
% 
%         if iter > 1 && errP(iter) <tol && ((norm((obj(iter)-obj(iter-1)),'fro'))^2/(norm(obj(iter-1),'fro'))^2)<tol
%             break;
%         end        
%     end       
   
%     fprintf('Update G, Time:%g\n', etime(clock,t1));

end

