from keras.preprocessing.image import ImageDataGenerator
import os

batch_size = 4




def main():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    print('test')

    i=0
    for batch in datagen.flow_from_directory(
            'dataset',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary',
            save_to_dir='test'):

        i += 1
        if i>9:
            break
        print('testI')


if __name__ == '__main__':
    main()
