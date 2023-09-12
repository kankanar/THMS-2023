load('GTD.mat'); % Please write your file name for hand bounding boxes in Matlab format.
addpath(genpath('./code'));
for i=1:height(T)
    imname = T.imageFilename(i);
    imname = split(imname,'\');
    imnam = fullfile('image',imname{end});%please insert folder path for images to be used here
    im = imread(imnam);
    label = zeros(size(im),'like',im);
    box = T.BBoxes{i};
    for j=1:size(box,1)
        x = round(box(j,1)+box(j,3)/2);
        y = round(box(j,2)+box(j,4)/2);
        if(box(j,1)<=0)
            box(j,1) = 1;
        end
        if(box(j,2)<=0)
            box(j,2) = 1;
        end
        label(box(j,2):box(j,2)+box(j,4)-5,box(j,1):box(j,1)+box(j,3)-5,:) = 128;
        label(y-6:y+6,x-6:x+6,:) = 255;
    end
    mask=zeros(size(label,1),size(label,2));
    fore=(label(:,:,1)==255);
    back=(label(:,:,1)==0);

    mask(fore)=1;
    mask(back)=-1;
   [alpha]=learningBasedMatting(im,mask);
   imwrite(uint8(imbinarize(alpha)*255),fullfile('semilabel',imname{end})); %please insert save folder name here
end
