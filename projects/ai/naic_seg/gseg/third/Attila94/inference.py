import os, cv2
import numpy as np
from time import localtime, strftime

from keras.preprocessing.image import ImageDataGenerator
from model.refinenet import build_refinenet
from scripts.helpers import get_label_info, saveResult

weights = 'model/refinenet.hdf5'
input_shape = (512,1024,3) # height x width x channels

# Define parameters
input_dir = 'path_to_input'
class_dict = 'path/class_dict.csv'

def preprocImage(img):
    batch = img[:,:,:,::-1] # RGB to BGR
    batch[:,:,:,0] -= 103.939
    batch[:,:,:,1] -= 116.779
    batch[:,:,:,2] -= 123.68
    return batch
    
def testGenerator(test_path,
                  target_size,
                  batch_size,
                  out_dir = None):
    '''
    Data generator for testing.
    
    Arguments: 
        test_path: Path to directory containing test images.
        target_size: Size to which input images are resized before being fed
            into RefineNet.
        batch_size: Images per batch.
        out_dir: Path to output directory.
    '''
    image_datagen = ImageDataGenerator()
    
    image_generator = image_datagen.flow_from_directory(
        os.path.dirname(test_path),
        classes = [os.path.basename(test_path)],
        class_mode = None,
        target_size = target_size,
        batch_size = batch_size,
        shuffle = False)

    for batch in image_generator:
        idx = ((image_generator.batch_index - 1)%len(image_generator)) * image_generator.batch_size
        file_names = image_generator.filenames[idx : idx + image_generator.batch_size]
        if out_dir:
            for img, file_name in zip(batch,file_names):
                cv2.imwrite(os.path.join(out_dir,os.path.basename(file_name)), img[:,:,::-1])
        batch = preprocImage(batch)
        if image_generator.batch_index == 0:
            yield batch, file_names
            return
        yield batch, file_names

# Generate output directories
output_dir = os.path.join('predictions',strftime("%Y%m%d-%H%M%S", localtime()))
if not os.path.exists(output_dir):
    org_dir = os.path.join(output_dir,'input') # input images save dir
    pred_dir = os.path.join(output_dir,'pred') # output predictions save dir
    os.makedirs(org_dir)
    os.makedirs(pred_dir)

# Create a text file containing the path to the weights used
with open(os.path.join(output_dir,'settings.txt'), 'w') as f:
    f.write('Weights: {}\n'.format(weights))

# Import classes from csv file
class_names_list, mask_colors, num_class, class_names_string = get_label_info(class_dict)

# Define model and load weights
model = build_refinenet(input_shape, num_class)
model.load_weights(weights)

myTestGenerator = testGenerator(input_dir, input_shape[:2], 2, out_dir = org_dir)
for batch, file_names in myTestGenerator:
    results = model.predict_on_batch(batch)
    saveResult(results, pred_dir, file_names, mask_colors)