class Config:
    # vgg16 takes 224 as input
    width = 224
    height = 224 #round(1.5*width)

    classes = 12

    dataset = 'ds_normal'

    epochs = 10
    batch_size = 16

    #needs to be an int
    steps = 2000/batch_size

    #add model in elif struct in start_train
    #model 1, model 2, model 3
    model = 'model 1'

