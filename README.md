# p7_final
# RAW and RGB Image Classification Comparison

## Comparison RAW and RGB dataset

The dataset can be found within the 'Dataset' directory 

There are 21.017 RAW and RGB object sample pairs in the dataset divided into 5 classes, each with a resolution of 40x40 pixels

## Code Implementation

In the 'Segmentation' directory, the sample processing pipeline is shown for calculating each 40x40 object sample from a high resolution capture 550cm above a capture surface with rice grains.
In the 'SDK' directory, the C++ implementation of the Phase One Image SDK is shown for processing the IIQ 16-L 151 MP RAW images into binary files for the RAW data and into PNG files for the RGB data. 
In the 'BCA' directory, the tensorflow implementation code for combining original RAW images and 'packed' RAW images is shown. 
In the 'ResNet' directory, the tensorflow implementation code for the ResNet-101 convolutional neural network is shown.

## Acknowledgement
For collaboration and loan of XF IQ4 camera and Schneider Kreuznach LS 80mm f/2.8 lens, special thanks to Lau Nørgaard, Chief Technology Officer at Phase One. For assistance with the capture setup for the dataset, special thanks to Kenneth Knirke and Claus Vestergaard Skipper, Assistant Engineers, Department of Electronic Systems at AAU.
