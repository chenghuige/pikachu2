from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os, sys
from tqdm import tqdm
from tools.dataloader import test_data_generator
import config
from tools.layers import GlobalAveragePooling2D

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = config.Config('test')


def predict_single(input_path, output_path, model, n_class):

    for image in tqdm(os.listdir(input_path)):
        index, _ = os.path.splitext(image)
        img = cv2.imread(os.path.join(input_path, image), cv2.IMREAD_UNCHANGED)
        img = np.float32(img) / 127.5 - 1
        pr = model.predict(np.expand_dims(img, axis=0), verbose=1)[0]
        pr = pr.reshape((256, 256, n_class)).argmax(axis=2)
        seg_img = np.zeros((256, 256), dtype=np.uint16)
        for c in range(n_class):
            seg_img[pr[:, :] == c] = int((c + 1) * 100)
        cv2.imwrite(os.path.join(output_path, index + ".png"), seg_img)


def predict_batch(input_path, output_path, model, n_class):

    g = test_data_generator(input_path, cfg.batch_size)
    for x, r in g:
        out = model.predict(x, verbose=0)
        for i in range(out.shape[0]):
            pr = out[i].reshape((256, 256, n_class)).argmax(axis=2)
            seg_img = np.zeros((256, 256), dtype=np.uint16)
            for c in range(n_class):
                seg_img[pr[:, :] == c] = int((c + 1) * 100)
            cv2.imwrite(os.path.join(output_path, r[i].replace('tif', 'png')), seg_img)


if __name__ == "__main__":
    weights_path = cfg.weight_path
    input_path = cfg.data_path
    output_path = cfg.output_path
    n_class = cfg.n_classes
    cfg.check_folder(output_path)

    if weights_path is None:
        print('weights_path  ERROR!')
        sys.exit()
    print(f'loaded : {weights_path}')
    co = {'GlobalAveragePooling2D' : GlobalAveragePooling2D}
    model = load_model(weights_path, custom_objects= co)
    predict_batch(input_path, output_path, model, n_class)