imdir = dir('image\*png');% please insert your image directory where data is stored in image/train/image and image/test/image 
labeldir = dir('label\*tif');% plaese insert your mask directory where data is stored in image/train/label and image/test/label 
resdir = 'data';
%idx = randperm(numel(imdir),round(0.2*numel(imdir)));
mkdir(fullfile(resdir,'test','image'))
mkdir(fullfile(resdir,'test','label'))
m = 0;n= 0;
for i=1:numel(imdir)
    im = imread(fullfile(imdir(i).folder,imdir(i).name));
    label = imread(fullfile(labeldir(i).folder,labeldir(i).name));
    label = imbinarize(label);

    imwrite(im,fullfile(resdir,'test','image',sprintf('%d.png',n)));
    imwrite(label,fullfile(resdir,'test','label',sprintf('%d.png',n)));
    n = n+1;

end
