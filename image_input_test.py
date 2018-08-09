from keras.preprocessing.image import ImageDataGenerator
from config import Config as C

batch_size = 30


def main():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    print('test')

    i=0
    for batch in datagen.flow_from_directory(
            'dataset',  # this is the target directory
            target_size=(C.height, C.width),  # (height, width)
            batch_size=batch_size,
            class_mode='binary',
            save_to_dir='test'):

        i += 1
        if i>9:
            break
        print('testI')


if __name__ == '__main__':
    main()
