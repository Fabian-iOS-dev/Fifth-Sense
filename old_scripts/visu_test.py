from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization
from config import Config as C
import numpy as np
from matplotlib import pyplot as plt
import cv2

def visualise_(batch):
    img = np.squeeze(batch, axis=0)
    print('Image Shape II:',img.shape)
    plt.imshow(img)
    plt.show()

def nice_printer(model, img):
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)
    conv_img = np.squeeze(conv_img, axis=0)
    print(conv_img.shape)
    conv_img = conv_img.reshape(conv_img.shape[:2])
    print(conv_img.shape)
    plt.imshow(conv_img)
    plt.show()


img = cv2.imread('testnote_nocrop.png')
print('Image Shape I:', img.shape)
# plt.imshow(img)
# plt.show()

model = Sequential()
model.add(Conv2D(3, (7,7), strides=(2,2), padding='same', input_shape=(img.shape), data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

nice_printer(model, img)