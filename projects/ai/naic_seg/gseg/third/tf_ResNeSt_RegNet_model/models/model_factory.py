from .ResNest import ResNest
from .ResNest_3D import ResNest3D
from .RegNet import RegNet
from .res34_DETR import DETR as res34_DETR
from .ResNest50_DETR import DETR as ResNest50_DETR
from .GENet import GENet

def get_model(model_name='ResNest50',input_shape=(224,224,3),n_classes=81,
                verbose=False,dropout_rate=0,fc_activation=None,**kwargs):
    '''get_model
    input_shape: (h,w,c)
    fc_activation: sigmoid,softmax
    '''
    model_name = model_name.lower()

    resnest_parameters = {
        'resnest50':{
            'blocks_set': [3,4,6,3],
            'stem_width': 32,
        },
        'resnest101':{
            'blocks_set': [3,4,23,3],
            'stem_width': 64,
        },
        'resnest200':{
            'blocks_set': [3,24,36,3],
            'stem_width': 64,
        },
        'resnest269':{
            'blocks_set': [3,30,48,8],
            'stem_width': 64,
        },
    }

    resnest3d_parameters = {
        'resnest50_3d':{
            'blocks_set': [3,4,6,3],
            'stem_width': 32,
        },
        'resnest101_3d':{
            'blocks_set': [3,4,23,3],
            'stem_width': 64,
        },
        'resnest200_3d':{
            'blocks_set': [3,24,36,3],
            'stem_width': 64,
        },

    }
    regnet_parameters={
        'regnetx400':{
            'stage_depth': [1,2,7,12],
            'stage_width': [32,64,160,384],
            'stage_G': 16,
            'SEstyle_atten': "noSE"
        },
        'regnetx1.6':{
            'stage_depth': [2,4,10,2],
            'stage_width': [72,168,408,912],
            'stage_G': 24,
            'SEstyle_atten': "noSE"
        },
        'regnety400':{
            'stage_depth': [1,3,6,6],
            'stage_width': [48,104,208,440],
            'stage_G': 16,
            'SEstyle_atten': "SE"
        },
        'regnety1.6':{
            'stage_depth': [2,6,17,2],
            'stage_width': [48,120,336,888],
            'stage_G': 24,
            'SEstyle_atten': "SE"
        },
        'regnet':{
            'stage_depth': None,
            'stage_width': None,
            'stage_G': None,
            'SEstyle_atten': None
        },
        
    }


    if model_name in resnest_parameters.keys():
        model = ResNest(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        blocks_set=resnest_parameters[model_name]['blocks_set'], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
        stem_width=resnest_parameters[model_name]['stem_width'], avg_down=True, avd=True, avd_first=False,**kwargs).build()
    
    elif model_name in regnet_parameters.keys():
        model = RegNet(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        stage_depth=regnet_parameters[model_name['stage_depth']],stage_width=regnet_parameters[model_name['stage_width']],\
            stage_G=regnet_parameters[model_name['stage_G']],SEstyle_atten=regnet_parameters[model_name['SEstyle_atten']],**kwargs).build()
    
    if model_name in resnest3d_parameters.keys():
        model = ResNest3D(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        blocks_set=resnest3d_parameters[model_name]['blocks_set'], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
        stem_width=resnest3d_parameters[model_name]['stem_width'], avg_down=True, avd=True, avd_first=False,**kwargs).build()

    elif model_name == 'res34_detr':
        model = res34_DETR(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation, **kwargs).build()

    elif model_name == 'resnest_detr':
            model = ResNest50_DETR(verbose=verbose, input_shape=input_shape,
            n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation, **kwargs).build()
    elif model_name == 'genet_light':
            model = GENet(verbose=verbose, model_name='light',input_shape=input_shape,
            n_classes=n_classes, fc_activation=fc_activation, **kwargs).build()
    elif model_name == 'genet_normal':
            model = GENet(verbose=verbose, model_name='normal',input_shape=input_shape,
            n_classes=n_classes, fc_activation=fc_activation, **kwargs).build()
    elif model_name == 'genet_large':
            model = GENet(verbose=verbose, model_name='large',input_shape=input_shape,
            n_classes=n_classes, fc_activation=fc_activation, **kwargs).build()
    else:
        raise ValueError('Unrecognize model name {}'.format(model_name))
    return model

if __name__ == "__main__":

    # model_names = ['ResNest50','ResNest101','ResNest200','ResNest269']
    model_names = ['ResNest50']
    # model_names = ['RegNetX400','RegNetX1.6','RegNetY400','RegNetY1.6']
    input_shape = [224,244,3]
    n_classes=81
    fc_activation='softmax' #softmax sigmoid

    for model_name in model_names:
        print('model_name',model_name)
        model = get_model(model_name=model_name,input_shape=input_shape,n_classes=n_classes,
                    verbose=True,fc_activation=fc_activation)
        print('-'*10)

    #RegNetY600 set
    # model = get_model(model_name="RegNet",input_shape=input_shape,n_classes=n_classes,
    #             verbose=True,fc_activation=fc_activation,stage_depth=[1,3,7,4],
    #             stage_width=[48,112,256,608],stage_G=16,SEstyle_atten="SE",active='mish')
    # print('-'*10)