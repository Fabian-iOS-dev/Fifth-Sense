from keras.preprocessing.image import ImageDataGenerator
import random
from config import Config as C

random.seed(300)

if C.model == 'model 1':
    from model_1 import model as m
    print('Using Tutorial')
elif C.model == 'model 2':
    from model_2 import model as m
    print('Using VGG16')
elif C.model == 'model 3':
    from model_3 import model as m
    print('Using Paper')
elif C.model == 'model 4':
    from model_4 import model as m
    print('Using Mobilenet')


def main():

    train_gen = ImageDataGenerator(
            rescale=1./255,
            # shear_range=0.2,
            # zoom_range=0.2,
            # width_shift_range=0.1,
            # height_shift_range=0.1,
            horizontal_flip=True,
            rotation_range=30
            )

    train_pics = train_gen.flow_from_directory(
            C.dataset + '/train',  # this is the target directory
            target_size=(C.height, C.width),  # all images will be resized
            batch_size=C.batch_size,
            class_mode='categorical',
            save_to_dir='sample', )

# save sample of pivture transformations
    # i = 0
    # for batch in train_gen.flow_from_directory(
    #         'dataset',  # this is the target directory
    #         target_size=(C.height, C.width),  # (height, width)
    #         batch_size=C.batch_size,
    #         class_mode='categorical',
    #         save_to_dir='sample'):
    #
    #     i += 1
    #     if i > 9:
    #         break

    test_gen = ImageDataGenerator(
            rescale=1./255,
            # horizontal_flip=True,
            # rotation_range=40
            )


    test_pics = test_gen.flow_from_directory(
            C.dataset + '/test',  # this is the target directory
            target_size=(C.height, C.width),  # all images will be resized to 150x150
            batch_size=C.batch_size,
            class_mode='categorical',
            save_to_dir='sample',
            )


    m.fit_generator( #training on train_data
            train_pics,
            steps_per_epoch=C.steps,
            epochs=C.epochs,
            validation_data=test_pics,
            # validation_steps=100 // C.batch_size,
            )

    # from keras.models import load_model
    # model = load_model('my_model.h5')

if __name__ == '__main__':
    main()