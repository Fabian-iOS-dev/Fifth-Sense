from keras import applications, optimizers
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from config import Config as C

#mobilenet

model_ = applications.mobilenet.MobileNet(weights="imagenet", include_top=False, input_shape=(C.width, C.height, 3),
                                  pooling='avg')
for layer in model_.layers[:10]:
    layer.trainable = False

x = model_.output
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=model_.input, outputs=predictions)

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.001), metrics=["accuracy"])