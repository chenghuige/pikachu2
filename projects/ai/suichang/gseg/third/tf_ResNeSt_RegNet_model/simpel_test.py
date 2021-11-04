from models.model_factory import get_model

if __name__ == "__main__":

    # model_names = ['ResNest50','ResNest101','ResNest200','ResNest269']
    # model_names = ['ResNest50']
    # model_names = ['resnest50_3d','resnest101_3d']
    model_names = ['GENet_light','GENet_normal','GENet_large']
    
    # model_names = ['RegNetX400','RegNetX1.6','RegNetY400','RegNetY1.6']
    # input_shape = [224,224,3]
    input_shape = [10,256,256,3]
    n_classes=81
    fc_activation='softmax' #softmax sigmoid

    # resnest
    for model_name in model_names:
        print('model_name',model_name)
        model = get_model(model_name=model_name,input_shape=input_shape,n_classes=n_classes,
                    verbose=True,fc_activation=fc_activation,using_cb=False)
        print('-'*10)

    #RegNetY600 set
    # model = get_model(model_name="RegNet",input_shape=input_shape,n_classes=n_classes,
    #             verbose=True,fc_activation=fc_activation,stage_depth=[1,3,7,4],
    #             stage_width=[48,112,256,608],stage_G=16,SEstyle_atten="SE",active='mish')
    # print('-'*10)

    #DETR
    # model_name = 'res34_DETR'
    # print('model_name',model_name)
    # model = get_model(model_name=model_name,input_shape=input_shape,
    #                   n_classes=n_classes,verbose=True,training=None,
    #                   fc_activation=fc_activation)
    # print('-'*10)

    # model_names = ['ResNest50_DETR','res34_DETR']
    # for model_name in model_names:
    #     print('model_name',model_name)
    #     model = get_model(model_name=model_name,input_shape=input_shape,
    #                   n_classes=n_classes,verbose=True,
    #                   fc_activation=fc_activation,using_cb=True)
    
