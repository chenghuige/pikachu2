import tensorflow as tf

# tf.enable_eager_execution()
# tf.compat.v1.enable_eager_execution()
from tensorflow.keras import models
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    MaxPooling2D,
    Conv1D,
)

from models.transformer import Transformer


def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


class Mish(Activation):
    '''
    based on https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
    Mish Activation Function.
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    # with tf.device("CPU:0"):
    result = inputs * tf.math.tanh(tf.math.softplus(inputs))
    return result


class DETR():
    def __init__(self, verbose=False, input_shape=(224, 224, 3), active="relu", n_classes=81,
                 dropout_rate=0.2, fc_activation=None,using_transformer=True,using_cb=None,
                 hidden_dim=512,nheads=8,num_encoder_layers=6,num_decoder_layers=6,n_query_pos=100):

        self.channel_axis = -1 #not for change
        self.verbose = verbose
        self.active = active #default relu
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.fc_activation = fc_activation

        self.using_transformer = using_transformer
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.n_query_pos = n_query_pos
        self.using_cb = using_cb #not imple yet


    def _make_stem(self,input_tensor,stem_width=64,deep_stem=False):
        x = input_tensor
        if deep_stem:
            x = Conv2D(stem_width,kernel_size=3,strides=2,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=False,data_format='channels_last')(x)

            x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(stem_width,kernel_size=3,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=False,data_format='channels_last')(x)

            x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(stem_width*2,kernel_size=3,strides=1,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=False,data_format='channels_last')(x)

            # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = Activation(self.active)(x)
        else:
            x = Conv2D(stem_width,kernel_size=7,strides=2,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=False,data_format='channels_last')(x)
            # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = Activation(self.active)(x)
        return x

    def _rsoftmax(self,input_tensor,filters,radix, groups):
        x = input_tensor
        batch = x.shape[0]
        if radix > 1:
            x = tf.reshape(x,[-1,groups,radix,filters//groups])
            x = tf.transpose(x,[0,2,1,3])
            x = tf.keras.activations.softmax(x,axis=1)
            x = tf.reshape(x,[-1,1,1,radix*filters])
        else:
            x = Activation('sigmoid')(x)
        return x

    def _make_block_basic(self,input_tensor,filters=64,kernel_size=3, stride=1,
                        conv_shortcut=True,mask=None):
        x = input_tensor
        preact = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        preact = Activation(self.active)(preact)

        if conv_shortcut is True:
            shortcut = Conv2D(filters, 1, strides=stride)(preact)
        else:
            shortcut = MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(preact)
        x = Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = Conv2D(filters, kernel_size, strides=1,
                        use_bias=False)(x)
        # if self.verbose: print('before add x ',x.shape)
        # if self.verbose: print('before add shortcut',shortcut.shape)
        # x = TimeDistributed(Add())([shortcut, x])
        x = Add()([shortcut, x])
        return x

    def _make_layer(self,input_tensor,filters=64,blocks=4,stride1=2,mask=None):
        x = input_tensor
        x = self._make_block_basic(x, filters, conv_shortcut=True)
        for i in range(2, blocks):
            x = self._make_block_basic(x, filters)
        x = self._make_block_basic(x, filters, stride=stride1)
        return x


    def get_trainable_parameter(self,shape=(100,128)):
        w_init = tf.random_normal_initializer()
        parameter = tf.Variable(
            initial_value=w_init(shape=shape,
                                dtype='float32'),
            trainable=True)
        return parameter

    def __make_transformer_top(self,x,verbose=False):
        h = Conv2D(self.hidden_dim,kernel_size=1,strides=1,
                    padding='same',kernel_initializer='he_normal',
                    use_bias=True,data_format='channels_last')(x)
        if verbose: print('h',h.shape)

        if tf.__version__ < "2.0.0":
            H,W = h.shape[1].value,h.shape[2].value
        else:
            H,W = h.shape[1],h.shape[2]
        if verbose: print('H,W',H,W)
        
        query_pos = self.get_trainable_parameter(shape=(self.n_query_pos, self.hidden_dim))
        row_embed = self.get_trainable_parameter(shape=(100, self.hidden_dim // 2))
        col_embed = self.get_trainable_parameter(shape=(100, self.hidden_dim // 2))

        cat1_col = tf.expand_dims(col_embed[:W], 0)
        cat1_col = tf.repeat(cat1_col, H, axis=0)
        if verbose: print('col_embed',cat1_col.shape)

        cat2_row = tf.expand_dims(row_embed[:H], 1)
        cat2_row = tf.repeat(cat2_row, W, axis=1)
        if verbose: print('row_embed',cat2_row.shape)

        pos = tf.concat([cat1_col,cat2_row],axis=-1)
        if tf.__version__ < "2.0.0":
            pos = tf.expand_dims(tf.reshape(pos,[pos.shape[0].value*pos.shape[1].value,-1]),0)
        else:
            pos = tf.expand_dims(tf.reshape(pos,[pos.shape[0]*pos.shape[1],-1]),0)

        h = tf.reshape(h,[-1, h.shape[1]*h.shape[2],h.shape[3]])
        temp_input = pos+h

        h_tag = tf.transpose(h,perm=[0, 2, 1])
        if verbose: print('h_tag transpose1',h_tag.shape)
        h_tag = Conv1D(query_pos.shape[0],kernel_size=1,strides=1,
                    padding='same',kernel_initializer='he_normal',
                    use_bias=True,data_format='channels_last')(h_tag)
        if verbose: print('h_tag conv',h_tag.shape)
        h_tag = tf.transpose(h_tag,perm=[0, 2, 1])
        if verbose: print('h_tag transpose2',h_tag.shape)

        query_pos = tf.expand_dims(query_pos,0)
        if verbose: print('query_pos',query_pos.shape)
        query_pos+=h_tag
        query_pos-=h_tag

        self.transformer = Transformer(
                        d_model=self.hidden_dim, nhead=self.nheads, num_encoder_layers=self.num_encoder_layers,
                        num_decoder_layers=self.num_decoder_layers)
        atten_out, attention_weights = self.transformer(temp_input, query_pos)
        return atten_out

    def build(self):
        get_custom_objects().update({'mish': Mish(mish)})

        input_sig = Input(shape=self.input_shape)

        x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input_sig)
        x = Conv2D(64,kernel_size=7,strides=2,
                       padding='same',kernel_initializer='he_normal',
                       use_bias=True,data_format='channels_last')(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = MaxPooling2D(3, strides=2)(x)

        x = self._make_layer(x, filters=64, blocks=3)
        if self.verbose: print('layer1',x.shape)
        x = self._make_layer(x, filters=128, blocks=4)
        if self.verbose: print('layer2',x.shape)
        x = self._make_layer(x, filters=256, blocks=6)
        if self.verbose: print('layer3',x.shape)
        x = self._make_layer(x, filters=512, blocks=3, stride1=1)
        if self.verbose: print('layer4',x.shape)

        x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        
        if self.using_transformer:
            x = self.__make_transformer_top(x,verbose=self.verbose)
        else:
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            if self.verbose: print('GlobalAveragePooling2D',x.shape)

        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate, noise_shape=None)(x)

        fc_out = Dense(self.n_classes,kernel_initializer='he_normal', use_bias=False)(x)
        if self.verbose: print('fc_out',fc_out.shape)

        if self.using_transformer:
            fc_out = tf.reduce_sum(fc_out, 1)
            if self.verbose: print('fc_out sum',fc_out.shape)

        if self.fc_activation:
            fc_out = Activation(self.fc_activation)(fc_out)
        
        model = models.Model(inputs=input_sig, outputs=fc_out)

        if self.verbose: print("res34_DETR builded with input {}, output{}".format(input_sig.shape,fc_out.shape))
        if self.verbose: print('-------------------------------------------')
        if self.verbose: print('')

        return model

