from tools import dataloader
from tensorflow.keras import optimizers
from tools.callbacks import LearningRateScheduler
from tools.learning_rate import lr_decays_func
from tools.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint
from builders import builder
import tensorflow as tf
import argparse
import os
from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cfg = Config('train')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Choose the semantic segmentation methods.', type=str, default='DeepLabV3Plus')
    parser.add_argument('--backBone', help='Choose the backbone model.', type=str, default='ResNet152')
    parser.add_argument('--num_epochs', help='num_epochs', type=int, default=cfg.epoch)

    parser.add_argument('--weights', help='The path of weights to be loaded.', type=str, default='weights')
    parser.add_argument('--lr_scheduler', help='The strategy to schedule learning rate.', type=str,
                        default='cosine_decay',
                        choices=['step_decay', 'poly_decay', 'cosine_decay'])
    parser.add_argument('--lr_warmup', help='Whether to use lr warm up.', type=bool, default=False)
    parser.add_argument('--learning_rate', help='learning_rate.', type=float, default=cfg.lr)

    args = parser.parse_args()
    return args


def train(args):
    filepath = "weights-{epoch:03d}-{val_loss:.4f}-{val_mean_iou:.4f}.h5"
    weights_dir = os.path.join(args.weights, args.backBone + '_' + args.model)
    cfg.check_folder(weights_dir)
    model_weights = os.path.join(weights_dir, filepath)

    # build the model
    model, base_model = builder(cfg.n_classes, (256, 256), args.model, args.backBone)
    model.summary()

    # compile the model
    #sgd = optimizers.SGD(lr=cfg.lr, momentum=0.9)
    nadam = optimizers.Nadam(lr=cfg.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=[MeanIoU(cfg.n_classes)])

    # checkpoint setting
    model_checkpoint = ModelCheckpoint(model_weights, monitor='val_loss', save_best_only=True, mode='auto')

    # learning rate scheduler setting
    lr_decay = lr_decays_func(args.lr_scheduler, args.learning_rate, args.num_epochs, args.lr_warmup)
    learning_rate_scheduler = LearningRateScheduler(lr_decay, args.learning_rate, args.lr_warmup, cfg.steps_per_epoch,
                                                    num_epochs=args.num_epochs, verbose=1)

    callbacks = [model_checkpoint]

    # training...
    train_set = dataloader.train_data_generator(cfg.train_data_path, cfg.train_label_path, cfg.batch_size,
                                                cfg.n_classes, cfg.data_augment)
    val_set = dataloader.val_data_generator(cfg.val_data_path, cfg.val_label_path, cfg.batch_size, cfg.n_classes)

    start_epoch = 0
    if os.path.exists(weights_dir) and os.listdir(weights_dir):
        a = sorted(file for file in os.listdir(weights_dir))
        model.load_weights(weights_dir + '/' + a[-1], by_name=True)
        # if load success, output info
        print('loaded :' + '-' * 8 + weights_dir + '/' + a[-1])
        start_epoch = int(a[-1][8:11])

    model.fit(train_set,
              steps_per_epoch=cfg.steps_per_epoch,
              epochs=args.num_epochs,
              callbacks=callbacks,
              validation_data=val_set,
              validation_steps=cfg.validation_steps,
              max_queue_size= cfg.batch_size,
              initial_epoch=start_epoch)

if __name__ == '__main__':
    args = args_parse()
    train(args)

