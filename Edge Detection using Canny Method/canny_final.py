"""
DATASET DOWNLOAD FROM KAGGLE
1. Go to your kaggle account, Scroll to API section and Click Expire API Token to remove previous tokens.
2. Click on Create New API Token - It will download kaggle.json file on your machine.
3. Run the following commands(Choose the kaggle.json file that you downloaded).
"""
# ! pip install -q kaggle
# from google.colab import files
# files.upload()
# ! mkdir ~/.kaggle
# ! cp kaggle.json ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json
# ! kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels
# ! unzip \*.zip

"""IMPORTING NECESSARY LIBRARIES"""
import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

import warnings
warnings.filterwarnings('ignore')

"""DEFINING PATH TO DATA-SET"""
xray_path = '/content/Lung Segmentation/CXR_png'

"""DATA EXTRACTION"""
xray_path_list = os.listdir(xray_path)

xray_images = []
for i in range(len(xray_path_list)):
    xray_image_path1 = os.path.join(xray_path, xray_path_list[i])

    xray_image_read = cv2.imread(xray_image_path1, cv2.IMREAD_GRAYSCALE)

    xray_image_final = cv2.resize(xray_image_read, (512, 512))

    xray_images.append(xray_image_final)

xray_images = np.array(xray_images)

"""CANNY-EDGE DETECTION"""
def plot_edges(image_og, image_edge):
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_og, cmap='Greys')
    plt.xlabel('Original Image')

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_edge, cmap='Greys')
    plt.xlabel('Canny Edges')

    plt.show()


def canny_edge(image_og):
    image_blur = cv2.GaussianBlur(image_og, (3, 3), cv2.BORDER_DEFAULT)
    image_edge = cv2.Canny(image_blur, 35, 80)

    plot_edges(image_og, image_edge)


for i in range(xray_images.shape[0]):
    canny_edge(xray_images[i])
