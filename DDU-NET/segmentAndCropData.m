maskdir = dir('semilabel'); %Please insert the segment folder from previous code 
maskdir(1:2) = [];
maindir = dir('image');
maindir(1:3) = [];
for i=1:numel(maskdir)
    label = imread(fullfile(maskdir(i).folder,maskdir(i).name));
    im = imread(fullfile(maindir(i).folder,maindir(i).name));
    label = imresize(label,[size(im,1),size(im,2)]);
    label = imbinarize(label);
    im = double(im).*label;
    stats = regionprops(label,'BoundingBox');
    vals=cell2mat({stats.BoundingBox}');
    vals = sortrows(vals,[3 4],{'descend'});
    bb = round(vals(1,:));
    imc = imcrop(im,bb);
    imwrite(uint8(imc),fullfile('cropped image',maindir(i).name)); % Please insert the folder name where cropped images will be stored
end
