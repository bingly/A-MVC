clear;

%% provide original multi-view data (mapped)
dataname='MSRC-v1';  %ORL_mtv extendyaleb MSRC-v1 COIL20_3VIEWS BBCSport EYaleB10_mtv Caltech101-20 scene15
load(strcat('./',dataname,'.mat'));
fprintf('start: %s\n',dataname);

%% special 
if(strcmp(dataname, 'Caltech101-20'))
    X = X';
end

%% get an un-matched multi-view samples
if (exist('Y'))
    gt = Y;
end

if (min(gt) == 0)
    gt = gt + 1;
end
%  X{1} = X1; X{2} = X2; X{3} = X3;
%  X{1} = X1'; X{2} = X2'; X{3} = X3'; X{4} = X4';

if (exist('fea'))
    X = fea;
end

cls_num = length(unique(gt)); %Y or gt
 K = size(X,2);
 for v=1:K
    [X{v}]=NormalizeData(X{v});
 end
 
[X,labels,mappings] = processData(X,gt);
 
vssc = zeros(1, K);  %silhouette
ch   = zeros(1, K);  %Calinski-Harabasz
for k = 1:K
    fea = X{k};
    clust = kmeans(fea,cls_num);
    s = silhouette(fea,clust);
    vssc(k) = mean(s);
    
    %ch
    evaluation = evalclusters(fea,"kmeans","CalinskiHarabasz","KList",cls_num);
    ch(k) = evaluation.CriterionValues;
end

if(size(X{1},2) ~= size(X{2},2))
    for k=1:K
        X{k} = X{k}';
    end
end

save(strcat(dataname,'_umt.mat'),'X', 'labels', 'gt', 'vssc', 'ch', 'mappings');

[vs, is] = max(vssc);
[vc, ic] = max(ch);
fprintf('vssc_i: %d; ch_i: %d\n',is, ic); 
