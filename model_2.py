from keras.models import Model
from keras.layers import Dense
from config import Config as C
from keras.applications.vgg16 import VGG16

#VGG16



# Generate a model with all layers (with top)
vgg16 = VGG16(weights='imagenet', include_top=True, input_shape=(3, C.height, C.width))

#Add a layer where input is the output of the  second last layer
x = Dense(C.classes, activation='softmax', name='predictions')(vgg16.layers[-2].output)

#Then create the corresponding model
model = Model(input=vgg16.input, output=x)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save('my_model.h5') #save model (architecture + weights + optimizer state)




