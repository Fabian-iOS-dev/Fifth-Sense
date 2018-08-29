from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dropout, Dense
from keras.optimizers import SGD
from config import Config as C
from keras.initializers import glorot_uniform
import random

seed = random.seed(7)

#custom

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(3, C.height, C.width), data_format='channels_first'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_initializer=glorot_uniform(seed)))
model.add(Dropout(0.5))
model.add(Dense(output_dim=C.classes, activation='softmax', kernel_initializer=glorot_uniform(seed)))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #try different optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
model.save('my_model.h5') #save model (architecture + weights + optimizer state)