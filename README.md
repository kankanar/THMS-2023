# THMS-2023
This is the code for my IEEE Transaction on Human-Machine Systems paper titled *"Hand segmentation with dense dilated U-Net and structurally incoherent non-negative matrix factorization based gesture recognition"*. I hope you find this useful

# Proposed Algorithm
The proposed algorithm in the paper *"Hand segmentation with dense dilated U-Net and structurally incoherent non-negative matrix factorization based gesture recognition"* has three major steps. The end goal of our algorithm is to perform hand gesture recognition. We identified the problem as a two-step problem. (I) Hand segmentation and (II) Hand gesture recognition. In the current literature, there are very few approaches that perform joint hand segmentation + gesture recognition. The main reason is being the dataset for hand gestures does not normally come with the segmentation mask. We tackle the first sub-problem as a semi-supervised hand segmentation approach considering that there is no segmentation mask available. This step is handled by learning-based matting. The code is also subdivided into three folders.
## Step 1: Learning Based Matting:
The code is developed from a very well-known work "Learning based digital matting". We use image matting to generate pseudo-labels from bounding box over hand regions which can be easily drawn or already available.

You need to have tight bounding boxes over hand regions for intended images to be processed in MATLAB format (Please check MATLAB documentation for this). By running the MATLAB code in **Image based matting** folder named `segmentAndCropData.m` you are going to generate the pseudo segmentation label for the next step. Check for the inline comments in the code where you have to give your folder path for images and generated segmentation labels.

## Step 2: Hand Segmentation: 



