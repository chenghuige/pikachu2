'''
Based on https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/
'''
import numpy as np
import os, csv
from time import localtime, strftime
from contextlib import redirect_stdout
import cv2

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        _ = next(file_reader) # remove header
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    
    class_names_string = ""
    for class_name in class_names:
        if not class_name == class_names[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name
    
    return label_values, len(label_values)
    
def labelVisualize(y_pred, mask_colors):
    """
    Convert prediction to color-coded image.
    """
    x = np.argmax(y_pred, axis=-1)
    colour_codes = np.array(mask_colors)
    img = colour_codes[x.astype('uint8')]
    return img

def saveResult(results, save_path, file_names, mask_colors):
    for img, file_name in zip(results, file_names):
        img = labelVisualize(img, mask_colors)
        cv2.imwrite(os.path.join(save_path, os.path.basename(file_name)), img[:,:,::-1])
        
def gen_dirs():
    """
    Generate directory structure for storing files produced during current run.
    """
    date_time = strftime("%Y%m%d-%H%M%S", localtime())
    run_dir = os.path.join('runs',date_time)
    summary_path = os.path.join(run_dir,'model_summary.txt')
    settings_path = os.path.join(run_dir,'settings.txt')
    epoch_log_path = os.path.join(run_dir,'epoch_log_'+date_time+'.csv')
    batch_log_path = os.path.join(run_dir,'batch_log_'+date_time+'.csv')
    weight_dir = os.path.join(run_dir,'weights')
    img_dir = os.path.join(run_dir,'images')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(weight_dir)
        os.makedirs(img_dir)
    dirs = {'date_time' : date_time,
            'run_dir' : run_dir,
            'summary_path' : summary_path,
            'settings_path' : settings_path,
            'epoch_log_path' : epoch_log_path,
            'batch_log_path' : batch_log_path,
            'weight_dir' : weight_dir,
            'img_dir' : img_dir}
    return dirs

def save_settings(settings_path,
                  summary_path,
                  frontend_trainable = None,
                  datagen_args = None,
                  batch_size = None,
                  input_shape = None,
                  dataset_basepath = None,
                  frontend_weights = None,
                  steps_per_epoch = None,
                  epochs = None,
                  pre_trained_weights = None,
                  model = None,
                  lr_init = None):
    """
    Save summary of training settings used in current run.
    """
    
    with open(summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    with open(settings_path, 'w') as f:
        f.write('ResNet weights trainable: {}\n'.format(frontend_trainable))
        f.write('Data augmentation settings: {}\n'.format(datagen_args))
        f.write('Batch size: {}\n'.format(batch_size))
        f.write('Input shape: {}\n'.format(input_shape))
        f.write('Dataset path: {}\n'.format(dataset_basepath))
        f.write('ResNet weights path: {}\n'.format(frontend_weights))
        f.write('Steps per epoch: {}\n'.format(steps_per_epoch))
        f.write('Epochs: {}\n'.format(epochs))
        f.write('Pre-trained weights: {}\n'.format(pre_trained_weights))
        f.write('Initial learning rate: {}\n'.format(lr_init))