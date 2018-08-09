from keras.preprocessing.image import ImageDataGenerator
import random
from config import Config as C

random.seed(300)


if C.model == 'model_I':
    import model_I as m
elif C.model == 'model':
    import model as m

def main():


    train_gen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,)

    train_pics = train_gen.flow_from_directory(
            C.dataset + '/train',  # this is the target directory
            target_size=(C.height, C.width),  # all images will be resized
            batch_size=C.batch_size,
            class_mode='categorical')

    test_gen = ImageDataGenerator(
            rescale=1./255)

    test_pics = test_gen.flow_from_directory(
            C.dataset + '/test',  # this is the target directory
            target_size=(C.height, C.width),  # all images will be resized to 150x150
            batch_size=C.batch_size,
            class_mode='categorical')


    m.model.fit_generator( #training on train_data
    train_pics,
            steps_per_epoch=944 // C.batch_size,
            epochs=10,
            validation_data=test_pics,
            validation_steps=236 // C.batch_size,
            )

    # from keras.models import load_model
    # model = load_model('my_model.h5')

if __name__ == '__main__':
    main()