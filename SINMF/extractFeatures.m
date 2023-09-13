maindir = dir('train_cropped_images');
maindir(1:2) = [];
X = cell(1);
Y = cell(1);
k = 1;
p = 1;
load('trainedVGG16.mat');
inputSize = netTransfer.Layers(1).InputSize;
featurelayer = 'relu7';

    for i=1:numel(maindir)
    disp(i)
    subdir = dir(fullfile(maindir(i).folder,maindir(i).name,'*png'));
    human = zeros(inputSize(1),inputSize(2), inputSize(3), numel(subdir));
    for j=1:numel(subdir) 
        im = imread(fullfile(subdir(j).folder,subdir(j).name));
        human(:,:,:,j) = single(imresize(im,inputSize(1:2)));
    end
        hfeatures = activations(netTransfer,gather(human),featurelayer,'MiniBatchSize',64,'ExecutionEnvironment','gpu');
        hfeatures = gpuArray(hfeatures);
        hfeatures = gather(reshape(hfeatures,4096,size(hfeatures,4)));
        
        X(p,1) = {double(hfeatures)};
        p = p+1;
    end
maindir = dir('test_cropped_images');
maindir(1:2) = [];
for i=1:numel(maindir)
    disp(i)
    subdir = dir(fullfile(maindir(i).folder,maindir(i).name,'*png'));
    human = zeros(inputSize(1),inputSize(2), inputSize(3), numel(subdir));
    for j=1:numel(subdir) 
        im = imread(fullfile(subdir(j).folder,subdir(j).name));
        human(:,:,:,j) = single(imresize(im,inputSize(1:2)));
    end
        hfeatures = activations(netTransfer,gather(human),featurelayer,'MiniBatchSize',64,'ExecutionEnvironment','gpu');
        hfeatures = gpuArray(hfeatures);
        hfeatures = gather(reshape(hfeatures,4096,size(hfeatures,4)));
        
        Y(k,1) = {double(hfeatures)};
        k = k+1;
end
save('ExtractedFeatures.mat','X','Y');
