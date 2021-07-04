"""
# DATASET DOWNLOAD FROM KAGGLE
1. Go to your kaggle account, Scroll to API section and Click Expire API Token to remove previous tokens.  
2. Click on Create New API Token - It will download kaggle.json file on your machine.
3. Run the following commands(Choose the kaggle.json file that you downloaded).
"""
#! pip install -q kaggle
#from google.colab import files
#files.upload()
#! mkdir ~/.kaggle
#! cp kaggle.json ~/.kaggle/
#! chmod 600 ~/.kaggle/kaggle.json
#! kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels
#! unzip \*.zip

""" IMPORTING NECESSARY LIBRARIES """
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import cv2

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
tf.config.run_functions_eagerly(True)

"""DEFINING PATH TO DATA-SET"""
image_path = '/content/Lung Segmentation/CXR_png'
mask_path = '/content/Lung Segmentation/masks'

"""DATA EXTRACTION AND PRE-PROCESSING"""
mask_path_list = os.listdir(mask_path)
image_path_list = [x.replace('_mask', '') for x in mask_path_list]

images = []
masks = []
for i in range(len(mask_path_list)):
    image_path1 = os.path.join(image_path, image_path_list[i])
    mask_path1 = os.path.join(mask_path, mask_path_list[i])

    image_read = cv2.imread(image_path1, 1)
    mask_read = cv2.imread(mask_path1, -1) 
    
    image_final = cv2.resize(image_read, (128, 128))        
    mask_final = cv2.resize(mask_read, (128, 128))
    
    images.append(image_final)
    masks.append(mask_final)

images = np.array(images)
masks = np.array(masks)

images = images/255
masks = masks/255

"""TRAIN/VALIDATION/TEST SPLIT"""
train_images, x_images, train_masks, y_masks = train_test_split(images, masks, test_size=0.272, random_state=0)
test_images, valid_images, test_masks, valid_masks = train_test_split(x_images, y_masks, test_size=0.5, random_state=0)

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
    
    inputs = Input((128, 128, 3))
    
    c1, p1 = down_samp(inputs, f[0])
    c2, p2 = down_samp(p1, f[1])
    
    bn = bottleneck(p2, f[2])
    
    d1 = up_samp(bn, c2, f[1])
    d2 = up_samp(d1, c1, f[0])
    
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d2)
  
    model = Model(inputs, outputs)
    
    return model

def dice_coefficient(y_true, y_pred):
  intersection = K.sum(y_true * y_pred, axis=[1,2])
  union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
  dice_coef = K.mean((2. * intersection)/union, axis=0)
  return dice_coef

model = unet()
model.compile(optimizer='adam', 
              loss = 'binary_crossentropy',
              metrics = [dice_coefficient])
model.summary()

"""TRAINING/SAVING THE MODEL"""
filepath = "/content/model_unet.h5"
checkpoint = ModelCheckpoint(filepath, monitor="val_dice_coefficient", save_best_only=True, mode='max', save_freq="epoch")
history = model.fit(train_images,
                    train_masks,
                    epochs=100,
                    callbacks=[checkpoint],
                    validation_data=(valid_images, valid_masks))

"""DICE COEFFICIENT AND LOSS PLOT"""
plt.plot(history.history['dice_coefficient'], label='Dice Coefficient (Train)')
plt.plot(history.history['val_dice_coefficient'], label = 'Dice Coefficient (Validation)')
plt.xlim([0, 100])
plt.ylim([0.30, 1])
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.title('Epochs vs Dice Coefficient')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.figure()

plt.plot(history.history['loss'], label='Loss (Train)')
plt.plot(history.history['val_loss'], label = 'Loss (Validation)')
plt.xlim([0, 100])
plt.ylim([0, 0.48])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epochs vs Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

"""LOAD THE BEST SAVED MODEL"""
model_load = load_model(filepath, custom_objects={"dice_coefficient": dice_coefficient})
model_load.compile(optimizer='adam', 
              loss = 'binary_crossentropy',
              metrics = [dice_coefficient])
model_load.summary()

"""EVALUATION ON ON TEST DATA-SET"""
evalu_load = model_load.evaluate(test_images, test_masks, verbose=0)
print('Dice Coefficient of model on Test Data-Set = '"{:.2f}".format(evalu_load[1]*100), '%')

"""PREDICTION ON TEST DATA-SET"""
results_load1 = model.predict(test_images)
results_load2 = results_load1 > 0.5
results_load = results_load2.astype(int)

def plot_graphs(image, true, pred):
  plt.figure(figsize=(10,10))
  
  plt.subplot(1,3,1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image)
  plt.xlabel('Original X-Ray')
  

  plt.subplot(1,3,2)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(true, cmap='Greys')
  plt.xlabel('True Mask')

  plt.subplot(1,3,3)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow((tf.squeeze(pred)), cmap='Greys')
  plt.xlabel('Predicted Mask')

  plt.show()

for i in range(len(results_load)):
  plot_graphs(test_images[i], test_masks[i], results_load[i])
