# Medical Imaging and Machine Learning
Pixel-wise Image Segmentation of Biomedical Images using UNET Model and Tensorflow Library.

## Datset Used for Lung Sugmentation
https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels

* Windows user can use this command to rename files in masks folder - get-childitem *.png | foreach {rename-item $_ $_.name.replace("_mask", "")
