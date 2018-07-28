from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Flatten, Dropout, Dense
from start_train import height, width
from keras.applications.vgg16 import VGG16




# Generate a model with all layers (with top)
vgg16 = VGG16(weights=None, include_top=True)

#Add a layer where input is the output of the  second last layer
x = Dense(6, activation='softmax', name='predictions')(vgg16.layers[-2].output)

#Then create the corresponding model
model = Model(input=vgg16.input, output=x)
model.summary()




# model = Sequential()
#
# model.add(Conv2D(16, (3, 3), input_shape=(3, height, width)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2,2))
#
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(output_dim=6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save('my_model.h5') #save model (architecture + weights + optimizer state)




