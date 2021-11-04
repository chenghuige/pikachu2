# 源代码在Windows + CPU环境下使用，
# 若要使用GPU，可将train/predict中强制使用CPU的相关代码注释

# 文件说明
训练集保存在./train中，其中
(1)./train/images存放训练图像
(2)./train/labels存放人工标记

测试集保存在./test中，其中
(1)./test/images存放测试图像
(2)./test/labels存放人工标记

weight.h5是一个预训练的网络权重，精度有限，仅用于测试

model.py中使用tensorflow.keras定义了一个神经网络model.unet(n_class, l1_skip_conn=True)

# 模型训练  ——  train.py
训练模型调用train(
                model,
                image_folder,
                label_folder,
                n_class, 
                batch_size=4,
                epochs=4,
                weights_path = None)方法
其中:
        model:          传入的模型，可以使用定义好的model.unet(n_class)
        image_folder:   表示训练集图片文件夹，即./train/images
        label_folder:   表示训练集人工标记文件夹，即./train/labels
        n_class:        表示类别数量。例如，类别表示为[100,200,300]时n_class为3
        weights_path：  读取权重路径，默认为空不导入

调用示例：
u_model = model.unet(8)
image_folder = "./train/images/"
label_folder = "./train/labels/"
n_class = 8
weights_path = "./weights.h5"
train(u_model, image_folder, label_folder, n_class, weights_path)

# 模型预测  ——  predict.py
(1)对单张图片进行预测单张图片调用predict(image_file, name, model, output_path, n_class, weights_path=None)方法，
其中:
        image_file:     预测图片路径
        name:           保存图片名称
        model:          传入模型，可以使用定义好的model.unet(n_class)
        output_path:    预测图片保存文件夹，即./test/labels
        n_class：       类别数量
        weights_path：  读取权重路径，默认为空不导入

调用示例：
image_file = "./test/images/1.tif"
name = "predict"
u_model = model.unet(8)
output_path = "./test/labels/"
n_class = 8
weights_path = "./weights.h5"
predict(image_file, name, u_model, output_path, n_class, weights_path)

(2)对文件夹中的所有图片进行预测调用predict_all(input_path, output_path, model, n_class, weights_path)方法，
其中:   
        input_path:     预测图片所在的文件夹，即./test/images
        output_path：   预测图片保存文件夹，即./test/labels
        model:          传入模型，可以使用定义好的model.unet(n_class)
        n_class：       类别数量
        weights_path：  读取权重路径，默认为空不导入

调用示例：
input_path = "./test/images/"
u_model = model.unet(8)
output_path = "./test/labels/"
n_class = 8
weights_path = "./weights.h5"
predict_all(input_path, output_path, u_model, n_class, weights_path)
