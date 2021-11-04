import keras2onnx
import tensorflow as tf
from tensorflow import keras
import os
import argparse
import fast_scnn
import deeplabv3plus
import separable_unet
import onnx
import cityscapesscripts.helpers.labels as labels


def convert_logits_to_labels(x):
    """
    When this function receives a tensor it should be of
    shape [?, target_size[0], target_size[1], 20] and they
    should be the logits, not probabilities or classes.
    This function will be used in a lambda layer to convert these
    to the label ids given by cityscapes. These are given by the table below:

    | id | name                 |           color |
    -----------------------------------------------
    |  0 | unlabeled            |       (0, 0, 0) |
    |  1 | ego vehicle          |       (0, 0, 0) |
    |  2 | rectification border |       (0, 0, 0) |
    |  3 | out of roi           |       (0, 0, 0) |
    |  4 | static               |       (0, 0, 0) |
    |  5 | dynamic              |    (111, 74, 0) |
    |  6 | ground               |     (81, 0, 81) |
    |  7 | road                 |  (128, 64, 128) |
    |  8 | sidewalk             |  (244, 35, 232) |
    |  9 | parking              | (250, 170, 160) |
    | 10 | rail track           | (230, 150, 140) |
    | 11 | building             |    (70, 70, 70) |
    | 12 | wall                 | (102, 102, 156) |
    | 13 | fence                | (190, 153, 153) |
    | 14 | guard rail           | (180, 165, 180) |
    | 15 | bridge               | (150, 100, 100) |
    | 16 | tunnel               |  (150, 120, 90) |
    | 17 | pole                 | (153, 153, 153) |
    | 18 | polegroup            | (153, 153, 153) |
    | 19 | traffic light        |  (250, 170, 30) |
    | 20 | traffic sign         |   (220, 220, 0) |
    | 21 | vegetation           |  (107, 142, 35) |
    | 22 | terrain              | (152, 251, 152) |
    | 23 | sky                  |  (70, 130, 180) |
    | 24 | person               |   (220, 20, 60) |
    | 25 | rider                |     (255, 0, 0) |
    | 26 | car                  |     (0, 0, 142) |
    | 27 | truck                |      (0, 0, 70) |
    | 28 | bus                  |    (0, 60, 100) |
    | 29 | caravan              |      (0, 0, 90) |
    | 30 | trailer              |     (0, 0, 110) |
    | 31 | train                |    (0, 80, 100) |
    | 32 | motorcycle           |     (0, 0, 230) |
    | 33 | bicycle              |   (119, 11, 32) |
    | -1 | license plate        |     (0, 0, 142) |

    Any unlabeled by my training should be 19, thus these will turn into 0's
    which are ignored for evaluation. But this is for the Jetson so let's not worry about that
    here.
    """
    # Get the label Ids and use 0 as the void class
    CITYSCAPES_LABELS = [
        label for label in labels.labels if -1 < label.trainId < 255]
    CITYSCAPES_LABELS.append(labels.labels[0])
    VAL_IDS = [label.id for label in CITYSCAPES_LABELS]
    # Convert the IDs into a 2D tensor [20, 1]
    VAL_IDS = tf.convert_to_tensor(VAL_IDS, dtype=tf.int32)
    VAL_IDS = tf.reshape(VAL_IDS, (VAL_IDS.shape[0], 1))
    # Get the trainId labels
    x = tf.argmax(x, axis=-1)
    x = tf.expand_dims(x, -1)
    # Perform a one hot encode
    x = tf.one_hot(x, 20, axis=-1)
    # Remove the extra dimension
    x = tf.squeeze(x, -2)
    # Cast to int32
    x = tf.cast(x, tf.int32)
    # Do a matrix multiplication
    return tf.linalg.matmul(x, VAL_IDS)


def keras_to_onnx(model_choice, weights):
    """
    If this is being run on the Jetson
    Then Tensorflow 1.15.0 is recommended,
    keras2onnx 1.6 and onnxconverter-runtime 1.6 installed via pip.
    Its able to be converted.
    """
    if model_choice == 'fastscnn':
        model = fast_scnn.model(num_classes=20, input_size=(1024, 2048, 3))
        input_size = '1024x2048'
    elif model_choice == 'deeplabv3+':
        # Its important here to set the output stride to 8 for inferencing
        model = deeplabv3plus.model(num_classes=20, input_size=(
            1024, 2048, 3), depthwise=True, output_stride=8)
        input_size = '1024x2048'
    elif model_choice == 'separable_unet':
        # Was trained on a lower resolution
        model = separable_unet.model(num_classes=20, input_size=(512, 1024, 3))
        input_size = '512x1024'
    # Whatever the model is, load the weights chosen
    print("Loading the weights for {} at input size {}".format(
        model_choice, input_size))
    model.load_weights(weights)
    # Add the lambda layer to the model
    model = keras.Sequential([
        model,
        keras.layers.Lambda(convert_logits_to_labels)
    ])
    print("Weights Loaded")
    # Plot the model for visual purposes in case anyone asks what you used
    print("Plotting the model")
    tf.keras.utils.plot_model(
        model, to_file=os.path.join('./results', model_choice, model_choice+'.png'), show_shapes=True, dpi=300)
    # Convert keras model to onnx
    print("Converting Keras Model to ONNX")
    onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=8)
    # Get the filename
    onnx_filename = os.path.join(
        'results', model_choice, model_choice + '_' + input_size + '.onnx')
    # Save the ONNX model
    print("Saving the ONNX Model")
    onnx.save_model(onnx_model, onnx_filename)
    print("Conversion Complete, ready for Jetson AGX Xavier")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model to convert to uff",
                        choices=['fastscnn', 'deeplabv3+', 'separable_unet'])
    parser.add_argument("-w", "--weights-file", type=str, default="",
                        help="The weights the model will use in the uff file")
    args = parser.parse_args()
    # If its a file then use the keras_to_onnx
    if os.path.isfile(args.weights_file):
        keras_to_onnx(args.model, args.weights_file)
    # Else give an error
    else:
        parser.error("Please provide a weights file. File given not found.")
