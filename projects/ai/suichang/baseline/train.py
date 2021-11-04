from dataloader import image_segmentation_generator
from model import *

# 使用CPU
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(model, image_folder, label_folder, n_class, batch_size=4, epochs=4, weights_path=None):
    # model：传入模型
    # image_folder：图像文件夹
    # label_folder：分割数据文件夹
    # n_class：类别数量
    # weights_path：模型权重路径
    train_gen = image_segmentation_generator(
        image_folder, label_folder, batch_size, n_class)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    if weights_path != None:
        model.load_weights(weights_path)

    model.fit_generator(train_gen, 200, epochs=epochs)


def save_model(model, model_path):
    model.save_weights(model_path)


if __name__ == "__main__":
    weights_path = "../input/baseline/weight.h5"
    image_folder = "../input/baseline/train/images/"
    label_folder = "../input/baseline/train/labels/"
    n_class = 8

    model = unet(n_class)
    train(model, image_folder, label_folder,
          n_class, weights_path=weights_path)

    os.system('mkdir -p ../working/baseline')
    save_model(model, '../working/baseline/model.h5')
