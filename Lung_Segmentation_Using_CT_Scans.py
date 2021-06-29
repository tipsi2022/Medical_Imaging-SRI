"""
DATASET DOWNLOAD FROM KAGGLE
1. Go to your kaggle account, Scroll to API section and Click Expire API Token to remove previous tokens.  
2. Click on Create New API Token - It will download kaggle.json file on your machine.
3. Run the following commands(Choose the kaggle.json file that you downloaded).
"""
! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download -d tipsijadav/covid19-ct
! unzip \*.zip

"""IMPORTING NECESSARY LIBRARIES"""
import os
import cv2
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D

tf.config.run_functions_eagerly(True)

"""DEFINING PATH TO DATASET"""
image_path = '/content/ct_scans'
mask_path = '/content/lung_mask'

"""DATA EXTRCATION AND PRE-PROCESSING"""
def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return(array)

image_path_list = os.listdir(image_path)

images = []
masks = []
for row in image_path_list:
    image_path1 = os.path.join(image_path, row)
    mask_path1 = os.path.join(mask_path, row)

    image_read = read_nii(image_path1)
    mask_read = read_nii(mask_path1) 
    
    image_final = cv2.resize(image_read, (128, 128))        
    mask_final = cv2.resize(mask_read, (128, 128))
        
    for i in range(image_final.shape[2]):
        temp_image = np.expand_dims(image_final[...,i], axis=-1)
    
        images.append(temp_image)
        masks.append(mask_final[...,i])

images = np.array(images)
images = images/((images.max()) - (images.min()))

masks = np.array(masks, dtype='int64')
masks = tf.one_hot(masks, 3, dtype='float64', axis=-1).numpy()

"""TRAIN/VALIDATION/TEST SPLIT"""
x_train, test_images, y_train, test_masks = train_test_split(images, masks, test_size=0.2, random_state=0)

train_images, valid_images, train_masks, valid_masks = train_test_split(x_train, y_train, test_size=0.25, random_state=0)

"""CONVOLUTION BLOCKS"""
def down_samp(x, filters):
    c = Conv2D(filters, (3, 3), padding="same", strides=1, activation="relu", kernel_initializer='he_normal')(x)
    c = Conv2D(filters, (3, 3), padding="same", strides=1, activation="relu", kernel_initializer='he_normal')(c)
    p = MaxPooling2D((2, 2), (2, 2))(c)
    return c, p

def bottleneck(x, filters):
    c = Conv2D(filters, (3, 3), padding="same", strides=1, activation="relu", kernel_initializer='he_normal')(x)
    c = Conv2D(filters, (3, 3), padding="same", strides=1, activation="relu", kernel_initializer='he_normal')(c)
    return c

def up_samp(x, skip, filters):
    us = UpSampling2D((2, 2))(x)
    conc = Concatenate()([us, skip])
    c = Conv2D(filters, (3, 3), padding="same", strides=1, activation="relu", kernel_initializer='he_normal')(conc)
    c = Conv2D(filters, (3, 3), padding="same", strides=1, activation="relu", kernel_initializer='he_normal')(c)
    return c

"""UNET ARCHITECTURE"""
def unet():
    f = [8, 16, 32]
    
    inputs = Input((128, 128, 1))
    
    c1, p1 = down_samp(inputs, f[0])
    c2, p2 = down_samp(p1, f[1])
    
    bn = bottleneck(p2, f[2])
    
    d1 = up_samp(bn, c2, f[1])
    d2 = up_samp(d1, c1, f[0])
 
    outputs = Conv2D(3, (1, 1), padding="same", activation="softmax")(d2)
  
    model = Model(inputs, outputs)
    
    return model

def dice_coefficient(y_true, y_pred):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice_coef = K.mean((2. * intersection)/union, axis=0)
  return dice_coef

model = unet()
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',
              metrics = [dice_coefficient])
model.summary()

"""TRAINING THE MODEL"""
history = model.fit(train_images,
                    train_masks,
                    epochs=120,
                    validation_data=(valid_images, valid_masks))

"""DICE COEFFICIENT AND LOSS PLOT"""
plt.plot(history.history['dice_coefficient'], label='Dice Coefficient(train)')
plt.plot(history.history['val_dice_coefficient'], label = 'Dice Coefficient(validation)')
plt.xlim([0, 120])
plt.ylim([0.7, 1])
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.title('Epochs vs Dice Coefficient')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.figure()

plt.plot(history.history['loss'], label='Loss(train)')
plt.plot(history.history['val_loss'], label = 'Loss(validation)')
plt.xlim([0, 120])
plt.ylim([0, 0.45])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epochs vs Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

"""EVALUATION ON TEST DATA-SET"""

evalu = model.evaluate(test_images, test_masks, verbose=0)
print('Dice Coefficient of model on Test Data-Set = '"{:.2f}".format(evalu[1]*100), '%')