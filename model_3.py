from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dropout, Dense
from config import Config as C

#paper

model = Sequential()

#Conc1
model.add(Conv2D(96, (7,7), strides=(2,2), padding='same', input_shape=(3, C.height, C.width), data_format='channels_first'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

#Conv2
model.add(Conv2D(128, (5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

#Conv3
model.add(Conv2D(256, (3,3), padding='valid'))
model.add(Activation('relu'))

#Conv4
model.add(Conv2D(256, (3,3), padding='valid'))
model.add(Activation('relu'))

#Conv5
model.add(Conv2D(128, (3,3), strides=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
# model.output_shape()

#fc1
model.add(Flatten())
model.add(Dense(activation='relu', units=4096))

#fc2
model.add(Dense(activation='relu', units=2048))
# model.add(Dropout(0.65)) #removing this first to overfit model and then make it robust

#fc3
model.add(Dense(output_dim=C.classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#model.save('my_model.h5') #save model (architecture + weights + optimizer state)

