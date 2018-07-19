from keras.preprocessing.image import ImageDataGenerator
import model

batch_size = 16

def main():


    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
            'dataset/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')

    test_datagen = ImageDataGenerator(
            rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
    'dataset/test',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')


    model.model.fit_generator( #training on train_data
    train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)



    #what is faster?
    # model.save('my_model.h5') #save model (architecture + weights + optimizer state)
    # from keras.models import load_model
    # model = load_model('my_model.h5')



    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json) #architecture
    # model.save_weights("model.h5") #weigths


if __name__ == '__main__':
    main()