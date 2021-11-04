
import os
import argparse

import tensorflow as tf
from tensorflow.keras import backend as K
import uff
import fast_scnn
import deeplabv3plus
import separable_unet

dir = os.path.dirname(os.path.realpath(__file__))


def keras_to_uff(model_choice, weights):
    """
    This is important to READ.
    This must be done in a Tensorflow Version <2.
    In otherwords a Tensorflow version where graph computation was the norm.
    Due to the changes in Tensorflow 2, converting to uff is basically
    impossible to do since all the uff conversions now use DEPRECATED and REMOVED
    functions and classes. 
    If I'm honest, this is a PITA.

    Reference for this python file:
    https://devtalk.nvidia.com/default/topic/1028464/jetson-tx2/converting-tf-model-to-tensorrt-uff-format/
    """
    K.set_learning_phase(0)
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
    model.load_weights(weights)
    # Plot the model for visual purposes in case anyone asks what you used
    tf.keras.utils.plot_model(
        model, to_file=os.path.join('./results', model_choice, model_choice+'.png'), show_shapes=True)
    # Get the outputs
    outputs = []
    for output in model.outputs:
        outputs.append(output.name.split(":")[0])
    # Set the filename for the frozen graph
    frozen_graph = os.path.join(
        './results', model_choice, model_choice + '_' + input_size + '.pb')

    # Let's begin
    session = K.get_session()
    # Get the graph definition and remove training nodes, ignore deprecation warnings here...
    graph_def = tf.graph_util.convert_variables_to_constants(
        session, session.graph_def, outputs)
    graph_def = tf.graph_util.remove_training_nodes(graph_def)

    # Write frozen graph to file
    with open(frozen_graph, 'wb') as f:
        f.write(graph_def.SerializeToString())
    f.close()
    # Get the uff filename
    uff_filename = frozen_graph.replace('.pb', '.uff')
    # Convert and save as uff and we're done
    uff_model = uff.from_tensorflow_frozen_model(
        frozen_graph, outputs, output_filename=uff_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model to convert to uff",
                        choices=['fastscnn', 'deeplabv3+', 'separable_unet'])
    parser.add_argument("-w", "--weights-file", type=str, default="",
                        help="The weights the model will use in the uff file")
    args = parser.parse_args()
    if os.path.isfile(args.weights_file):
        keras_to_uff(args.model, args.weights_file)
    else:
        parser.error("Please provide a weights file. File given not found.")
