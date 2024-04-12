%% For convinience, we assume the order of the tensor is always 3;
clear;
addpath('tSVD','proxFunctions','solvers','twist', 'tools');
addpath('ClusteringMeasure', 'LRR', 'Nuclear_norm_l21_Algorithm', 'unlocbox');

dataname='MSRC_umt';  %ORL_umt  MSRC_umt COIL20_umt BBCSport_umt EYaleB10_umt scene15_umt
load(strcat('../data/',dataname,'.mat'));
fprintf('start: %s\n',dataname); 

%% preparation
cls_num = length(unique(labels{1}));
K = length(X); N = size(X{1},2); %sample number

[~, k_best] = max(ch);
[labels{k_best},index] = sort(labels{k_best});
X{k_best} = X{k_best}(:,index);
mappings{k_best} = sort(mappings{k_best});
omega = zeros(1, K); %weighted

alpha=0.0010;beta=1.0000;gamma=0.0001;theta=1.0000; %MSRC-v1
% alpha=0.0100;beta=0.0100;gamma=0.0010;theta=1.0000; %ORL
% alpha=1.0000;beta=1.0000;gamma=0.1000;theta=1.0000; %EYaleB10
% alpha=1.0000;beta=0.0010;gamma=0.1000;theta=0.0100; %COIL20
% alpha=0.0100;beta=0.0001;gamma=0.0010;theta=0.0100; %BBCSport
% alpha=1.0000;beta=0.0010;gamma=0.1000;theta=0.0010; %Scene15

epson = 1e-7;
max_mu = 1e10; pho_mu = 2;
max_iter = 100;

% initilize U, settings 
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';      % Binary  HeatKernel

for k=1:K   
    DX{k} = L2_distance_1(X{k},X{k});
    M{k} = eye(N,N);
    Z{k} = zeros(N,N); 
    G{k} = zeros(N,N);
	Y2{k} = zeros(N,N);
	
	%U
	TZ = constructW(X{k}',options);
	TZ = full(TZ);
	Z1 = TZ-diag(diag(TZ));         
	TZ = (Z1+Z1')/2;
	DZ= diag(sum(TZ));
	U{k} = DZ - TZ;                
    
    %tensor
    Y1{k} = zeros(N,N);  %Y1
    S{k} = zeros(N,N);  %S
    
    E{k} = zeros(size(X{k},1),N); 
    Y3{k} = zeros(size(X{k},1),N);  %Y3
    
    omega(k) = 10^(k-1);
%     omega(k) = 1;  %omega(k) = 10^(k-1);
end
clear TZ DZ Z1

omega = omega'; %omege needs convert to colomn vector

% initilize F
for k = 1:K
    sum_U = (U{k}+U{k}')*0.5;
    LUv = diag(sum(sum_U))-sum_U;
    LUv = (LUv+LUv')*0.5;
    try
        opts.tol = 1e-4; 
        [F{k},~] = eigs(LUv,cls_num,'sa',opts);   % U: n*num_cluster
    catch ME
        if (strcmpi(ME.identifier,'MATLAB:eig:NoConvergence'))
            opts.tol = 1e-4; 
            [F{k},~] = eigs(LUv, eye(size(LUv)),num_cluster,'sa',opts);
        else
            rethrow(ME);
        end
    end  
end


s = zeros(N*N*K,1);
y1 = zeros(N*N*K,1);

myNorm = 'tSVD_1';
sX = [N, N, K];
I_N = eye(N);

iter = 0;
mu = 1e-5; 

%calcute mapping acc, i.e., CPA@1 @3 @10
for k=1:K   
    a1{k} = [];
    a2{k} = [];
    a3{k} = [];
end

while(iter < max_iter)
%     fprintf('----processing iter %d--------\n', iter+1);
    for k=1:K
        %1 update Z^k
        TA = (mu*X{k}'*X{k} + 0.0001*I_N);  %优化
        TB = (2*gamma+mu)*M{k}*M{k}' + 0.0001*I_N;
        TC = (2*gamma*Z{k_best}+mu*(S{k}+Y1{k}/mu))*M{k}' + mu*X{k}'*(X{k}-E{k}+Y3{k}/mu);
        Z{k} = sylvester(TA,TB,TC);

        % Update M
%         M{k} = 1/(2*gamma+mu)*pinv(Z{k}'*Z{k})*Z{k}'*(2*gamma*Z{k_best}+mu*(S{k}+Y1{k}/mu));
        M{k} = 1/(2*gamma+mu)*inv(Z{k}'*Z{k}+0.0001*I_N)*Z{k}'*(2*gamma*Z{k_best}+mu*(S{k}+Y1{k}/mu));

        
        %2 update E^k
        TF = [];
        for k1=1:K
            TF = [TF;X{k1}-X{k1}*Z{k1}+Y3{k1}/mu];
        end
        [Econcat] = solve_l1l2(TF,beta/mu);
        %F = F';
        e_start = 1; e_end = 0;
        for k1 = 1:K
            e_end = e_end + size(X{k1},1);
            E{k1} = Econcat(e_start:e_end, :);
            e_start = e_start + size(X{k1},1);
        end
		
		%3 Update U^k
		DF = L2_distance_1(F{k}',F{k}');
		W = DX{k} + theta*DF;
		U{k} = Z{k} - (Y2{k}+W)/mu;
        for ic = 1:N
            idx    = 1:N;
            idx(ic) = [];
            U{k}(ic,idx) = EProjSimplex_new(U{k}(ic,idx));          % 
        end
            
        %4 update multipiers
        Y2{k} = Y2{k} + mu*(U{k}-Z{k});
        Y3{k} = Y3{k} + mu*(X{k}-X{k}*Z{k}-E{k});
    end
	
	%5 update  F
    for k = 1:K
        sum_U = (U{k}+U{k}')*0.5;
        LUv = diag(sum(sum_U))-sum_U;
        LUv = (LUv+LUv')*0.5;
        try
            opts.tol = 1e-4; 
            [F{k},~] = eigs(LUv,cls_num,'sa',opts);   % U: n*num_cluster
        catch ME
            if (strcmpi(ME.identifier,'MATLAB:eig:NoConvergence'))
                opts.tol = 1e-4; 
                [F{k},~] = eigs(LUv, eye(size(LUv)),num_cluster,'sa',opts);
            else
                rethrow(ME);
            end
        end  
    end
    
    %6 update  S
    for i=1:K
        Temp{i} = Z{k}*M{k};
    end
    
    Z_tensor = cat(3, Temp{:, :});
    Y_tensor = cat(3, Y1{:,:});
    z = Z_tensor(:);
    y1 = Y_tensor(:);
    
    %twist-version
    [s, objV] = wshrinkObj_weight(z - 1/mu*y1,omega*alpha/mu,sX,0,3);
%    [g, objV] = shrinkObj(z + (1/rho)*w,...
%                         1/rho,myNorm,sX,parOP);
       
    %% coverge condition 
    for k=1:K
        e1(k) = norm(X{k}-X{k}*Z{k}-E{k},inf);  
    end
    
    em1 = max(e1);
    fprintf('norm_Z %.8f    ', em1);
    
    S_tensor = reshape(s, sX);
    Z_tensor = reshape(z, sX);
    
    e2 = S_tensor - Z_tensor;
    em2 = max(abs(e2(:)));
    fprintf('S-Z: %7.10f    \n', em2);
    
    err = max([em1, em2]);
    if (err < epson)
        break;
    end

    iter = iter + 1;   
    % update Lagrange multiplier and  penalty parameter mu
    Y_tensor = Y_tensor + mu*e2;
    for i=1:K
        Y1{k} = Y_tensor(:,:,i);
        S{k} = S_tensor(:,:,i); 
    end
    mu = min(mu*pho_mu, max_mu);
    
    for k = 1: K
        a1{k} = [a1{k}, mappingsACC(M{k},mappings{k_best},1)];%CPA@1
        a2{k} = [a2{k}, mappingsACC(M{k},mappings{k_best},3)];%CPA@3
        a3{k} = [a3{k}, mappingsACC(M{k},mappings{k_best},10)];%CPA@10
    end
end
Affinity = 0;
for k=1:K
    Affinity = Affinity + abs(Z{k}*M{k})+abs(M{k}'*Z{k}');
end
gt = labels{k_best};

% Y = tsne(Affinity,'Algorithm','exact','Distance','seuclidean'); 
% gscatter(Y(:,1), Y(:,2),gt);

% figure(1); imagesc(S);
% S_bar = CLR(S, cls_num, 0, 0 );
% figure(2); imagesc(S_bar);
clu = SpectralClustering(Affinity,cls_num);
[A nmi avgent] = compute_nmi(gt,clu);
ACC = Accuracy(clu,double(gt));
[f,p,r] = compute_f(gt,clu);
[AR,RI,MI,HI]=RandIndex(gt,clu);

% fprintf('\n%.4f %.4f %.4f %.4f %.4f %.4f\n',ACC,nmi,AR,f,p,r);
fprintf('alpha=%.4f,beta=%.4f,gamma=%.4f,theta=%.4f: %.4f %.4f %.4f %.4f %.4f %.4f\n',alpha,beta,gamma,theta,ACC,nmi,AR,f,p,r);

% plot the graph of the alignment rate
iterations = 1:length(a1{1}); 
for k = 1:K
    if k == k_best
        continue
    end
    CPA_at_1 = a1{k}; 
    CPA_at_3 = a2{k}; 
    CPA_at_10 = a3{k}; 

    % 绘制折线图
    figure; % 创建一个新的图形窗口
    plot(iterations, CPA_at_1, '-c', 'LineWidth', 1.2); 
    hold on; % 保持当前图形，以便在同一个图上绘制更多线条
    plot(iterations, CPA_at_3, '--b', 'LineWidth', 1.2); 
    plot(iterations, CPA_at_10, '-*m', 'LineWidth', 1.2); 

    % 设置图例和标签
    set(gca,'FontSize',14) %字体
    legend('CPA@1', 'CPA@3', 'CPA@10');
    xlabel('Number of Iterations');
    ylabel(strcat('CPA@q of M^{(',num2str(k),')}'));
    title('MSRC-v1');
    
    hold off; % 释放图形，以便可以进行其他绘图操作
end