from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
from config import Config as C
from keras.applications.vgg16 import VGG16
from numpy import random

#VGG16

seed = random.seed(300)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(3, C.height, C.width))

# Generate a model with all layers (with top)
output_model = Sequential()
output_model.add(Flatten())
output_model.add(Dense(4096, activation='relu'))
output_model.add(Dense(C.classes, activation='softmax', name='predictions'))


#Then create the corresponding model
# this works but layers of output model are not shown
# model = Model(input=vgg16.input, output=output_model(vgg16.output))

model = Sequential() #model is the vgg16 architecture with modified output
for l in vgg16.layers:
    model.add(l)

for l in output_model.layers:
    model.add(l)

for layer in model.layers[:16]:
    layer.trainable = False

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save('my_model.h5') #save model (architecture + weights + optimizer state)