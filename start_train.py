from keras.preprocessing.image import ImageDataGenerator
import model
import random

random.seed(300)
batch_size = 16
width = 300
height = 300
dataset = 'normal_ds'

def main():


    train_gen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True)

    train_pics = train_gen.flow_from_directory(
            dataset + '/train',  # this is the target directory
            target_size=(height, width),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')

    test_gen = ImageDataGenerator(
            rescale=1./255)

    test_pics = test_gen.flow_from_directory(
            dataset + '/test',  # this is the target directory
            target_size=(height, width),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')


    model.model.fit_generator( #training on train_data
    train_pics,
            steps_per_epoch=944 // batch_size,
            epochs=10,
            validation_data=test_pics,
            validation_steps=236 // batch_size,
            )

    # from keras.models import load_model
    # model = load_model('my_model.h5')

if __name__ == '__main__':
    main()