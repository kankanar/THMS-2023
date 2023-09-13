load('ExtractedFeatures.mat')

for i=1:numel(X)
    A = X{i};
    [W,H]  = nnmf(A,10);
    C(i) = {W};
end
for i=1:numel(X)
    A = X{i};
    [W,H]  = structural_nmf(A,C,10,i);
    B(i) = {pinv(W)};
    C(i) = {W};
end
gtlabel = [];
reslabel = [];
p=1;
tic
for i=1:numel(Y)
    D = Y{i};
    for j=1:size(D,2)
        E = cell(1);
        feat = D(:,j);
        for k=1:numel(B)
                h = max(0,B{k}*feat);
                recim = C{k}*h;
                E(k) = {mse(feat,recim)};
         end
         gtlabel(p) = i;
         [~,idx] = min(cell2mat(E));
         reslabel(p) = idx;
         p = p+1;
            
    end
end
toc
acc = sum(gtlabel==reslabel)/numel(gtlabel)