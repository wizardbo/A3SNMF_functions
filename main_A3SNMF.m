function main_A3SNMF(Y,S,C,BOW,parameters,result_pth)
    n = length(Y);
    l = max(Y);

    alpha = 1;
    
    fid = fopen( result_pth,'wt');
    fprintf(fid,'%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\n', ... 
        'Dataset','num_nn','l','lambda1','lambda2','alpha','ACC','NMI','ARI','num_Times','Cost Time','Z_Norm','num_anomaly'); 
    
    dataname = parameters.dataname;
    parameters.num_cluster = l;
    parameters.times = 1;
    
    n_times = parameters.times;    
    for num_nn = [100] 
        new_S = replace_S_with_nn_neighbors(S, num_nn);     
%             WMD_S = replace_S_with_nn_neighbors(WMD_S, num_nn);  
        for lambda1 = [10]
            for lambda2 = [4:9]*0.1
                sum_results = zeros(parameters.times, 3);
                sum_time = 0;
                for t = 1:1
                    fprintf("%s ---- num_nn:%f_(%d), lambda1:%f, lambda2:%f, alpha:%f | \n", dataname, num_nn, t, lambda1, lambda2, alpha);
                    fprintf("----------------------------------------\n");

%                             idx = kmeans(DE,parameters.num_cluster);
%                             C = sparse(1:n,idx,1);  %sparse(i,j,v); 
                    [~,prec_label] = maxk(C,1,2); 
                    results = get_all_metrics(Y,prec_label');     
                    fprintf("Init result -- Acc:%.2f, Nmi:%.2f, Ari:%.2f\n", results.acc, results.nmi, results.ari);

                    tic;
                    t1 = clock;                            
          % ??init?? --------------------------------------------------------------------
                    [Z,V,D,G,new_C] = init_matrix(C,new_S);

%                             f = fopen( ['Results_new/' char(parameters.dataname) '_' char(parameters.method) '_' char(num2str(lambda2)) '_' char(num2str(lambda3)) '_' char(num2str(lambda4)) '.txt'], 'wt');                                                       
          % ??main function?? -----------------------------------------------------------
                    [new_C,zero_idxs] = A3SNMF(Y,new_S,Z,D,V,new_C,alpha,lambda1,lambda2,n,l);
                    fprintf("num_allzero:%d |\t", length(zero_idxs)); 
%                             fprintf(f, "num_allzero:%d |\t", length(zero_idxs));   

                    nnz_idxs = setdiff(1:n, zero_idxs);
                    fprintf("num_nnz:%d | percent_of_nnz:%f \n", length(nnz_idxs), length(nnz_idxs)/n); 
%                             fprintf(f, "num_nnz:%d | percent_of_nnz:%f \n", length(nnz_idxs), length(nnz_idxs)/n);   

          % ??predict Y?? ---------------------------------------------------------------
%                             fprintf(f, "gnd number: %d, init cluster number: %d\n", max(Y), size(C,2));   
                    prec_label = predict_Y(new_C, BOW, zero_idxs);
%                             fclose(f);

          % ??evaluate Yp?? -------------------------------------------------------------
                    results = get_all_metrics(Y,prec_label');

                    sum_results(t,1) = results.acc;
                    sum_results(t,2) = results.nmi;
                    sum_results(t,3) = results.ari;

                    time = etime(clock,t1);
                    sum_time = sum_time + time;

%                     fprintf("times-%d Results -- Acc:%.2f, Nmi:%.2f, Ari:%.2f\n", t, results.acc, results.nmi, results.ari);
%                     fprintf("----------------------------------------\n");
%                 
                    fprintf("Cost:%f \t",time);
                end
%                         mean_results = mean(sum_results,1);
%                         std_results = std(sum_results,1);

%                         Acc = mean_results(1);
%                         Nmi = mean_results(2);
%                         Ari = mean_results(3);
                Acc = sum_results(1);
                Nmi = sum_results(2);
                Ari = sum_results(3);
                avg_time = sum_time/n_times;

                fprintf("Final Results -- Acc:%.2f, Nmi:%.2f, Ari:%.2f\n", Acc, Nmi, Ari);
                fprintf("----------------------------------------\n");
                fprintf(fid,'%s\t %d\t %d\t %f\t %f\t %f\t %f\t %f\t %f\t %s\t %f\t %s\t %d\n', ...
                    dataname, num_nn, l, lambda1, lambda2, alpha, Acc, Nmi, Ari, "Final", avg_time,"Norm2", length(zero_idxs));
                % -----------------------------------------------------------------------
                if size(zero_idxs,2) > ceil(n * 0.7)
                    break;
                end                       
            end
        end

    end

    fclose(fid);
end

function new_S = replace_S_with_nn_neighbors(S, num_nn)
    n = size(S,1);

    % ??S???§Õ??§³????
    [~,I] = sort(S,2,'descend');    
    I = I(:,1:num_nn);    
    
    new_S = zeros(n,n);    
    for i = 1:n
        new_S(i,I(i,1:num_nn)) = S(i,I(i,1:num_nn));
    end
    new_S = new_S + (new_S==0).*new_S';
    
%     new_S = sparse(new_S);
end

function [Z,V,D,G,new_C] = init_matrix(C,new_S)
%     idx = kmeans(full(Em_BOW),l);
%     C = sparse(1:n,idx,1);  %sparse(i,j,v)
%     C = ones(n,l)/l;
%     C = matlab_gmm(Em_BOW(:,1:100), l);    

    n = size(C,1);
    
    new_C = C;
    V = C;
    Z = sparse(n,n);  
    G = new_S;  
    D = C;
end

function prec_label = predict_Y(C, BOW, zero_idxs)
    [n,l] = size(C);
    prec_label = zeros(n,1);
    
    nnz_idxs = setdiff(1:n, zero_idxs);

    [~,pre_nnz] = maxk(C(nnz_idxs,:),1,2);
    prec_label(nnz_idxs,1) = pre_nnz;

    if ~isempty(zero_idxs)
        spone_BOW = spones(sparse(BOW));
        d = exp(1)-1;

        centers = CFC(spone_BOW(nnz_idxs,:),prec_label(nnz_idxs,1),l,d);
        C(zero_idxs,:) = 1-pdist2(BOW(zero_idxs,:),centers,'cosine');
        C(isnan(C)) = 0; 

        [~,Ypre] = maxk(C(zero_idxs,:),1,2);
        prec_label(zero_idxs,1) = Ypre;
    end

    %  Print ??num_nnz per cluster??
    length_idxs = zeros(l,1);
    for i = 1:l
    %   i = l-j+1;
        length_idxs(i) = length(find(prec_label==i));
        fprintf("nnz: %d, total: %d documents in cluster %d\n", length(find(prec_label(nnz_idxs)==i)), length_idxs(i), i);
%         fprintf(fid, "nnz: %d, total: %d documents in cluster %d\n", length(find(prec_label(nnz_idxs)==i)), length_idxs(i), i);
    end

end

function results = prec_nnz_label(C, Y, zero_idxs)
% -------------------------- ???????????????? --------------------------------
    nnz_idxs  = find(any(C,2));
   % zero_idxs = setdiff(1:n, nnz_idxs);
    fprintf("num_allzero_C:%d |\t", length(zero_idxs));

    [~,prec_label] = maxk(C(nnz_idxs,:),1,2); 
    results = get_all_metrics(Y(nnz_idxs),prec_label');
end
