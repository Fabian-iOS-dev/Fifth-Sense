class Config:
    # vgg16 takes 224 as input
    width = 200
    height = 200

    classes = 6

    dataset = 'ds_normal'
    batch_size = 16

    #add model in elif struct in start_train
    #model, model_I
    model = 'model_I'

