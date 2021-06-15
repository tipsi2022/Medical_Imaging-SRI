# Medical Imaging and Machine Learning
Pixel-Wise Image Segmentation of Biomedical Images using UNET Model and Tensorflow Library.

## Lung Sugmentation using X-ray Images Dataset:-
* Link to Dataset -> https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels

* Windows user can use this command to rename files in masks folder -> get-childitem *.png | foreach {rename-item $_ $_.name.replace("_mask", "")

## Lung Sugmentation using CT Scans Dataset
* Link to Dataset -> https://www.kaggle.com/andrewmvd/covid19-ct-scans

* Rename the .nii files in ct_scans folders
