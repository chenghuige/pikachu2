#----------IMPORT PYTHON PACKAGES----------#
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import horovod.tensorflow.keras as hvd
from tensorflow import keras
import tensorflow as tf
import cityscapesscripts.helpers.labels as labels
import numpy as np
import matplotlib.image as Image
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import os

#----------IMPORT MODELS AND UTILITIES----------#
import unet
import separable_unet
import bayes_segnet
import deeplabv3plus
import fast_scnn
import lrfinder
import datasets

#----------CONSTANTS----------#
CITYSCAPES_LABELS = [
    label for label in labels.labels if -1 < label.trainId < 255]
# Add unlabeled
CITYSCAPES_LABELS.append(labels.labels[0])
# Get all the colors
CITYSCAPES_COLORS = [label.color for label in CITYSCAPES_LABELS]
# Get the IDS
CITYSCAPES_IDS = [label.trainId for label in CITYSCAPES_LABELS]
# Change unlabeled id from 255 to 19
CITYSCAPES_IDS[-1] = 19
CLASSES = 20

#----------ARGUMENTS----------#
parser = argparse.ArgumentParser(
    prog='train.py', description="Start training a semantic segmentation model")
parser.add_argument(
    "-m", "--model",
    help="Specify the model you wish to use: OPTIONS: unet, separable_unet, bayes_segnet, deeplabv3+, fastscnn",
    choices=['unet', 'bayes_segnet', 'deeplabv3+',
             'fastscnn', 'separable_unet'],
    required=True)
parser.add_argument(
    "-r", "--resume",
    help="Resume the training, specify the weights file path of the format weights-[epoch]-[val-acc]-[val-loss]-[datetime].hdf5")
parser.add_argument(
    '-p', "--path",
    help="Specify the root folder for cityscapes dataset, if not used looks for CITYSCAPES_DATASET environment variable")
parser.add_argument(
    '-c', "--coarse",
    help="Use the coarse images", action="store_true")
parser.add_argument(
    '-t', "--target-size",
    help="Set the image size for training, should be a elements of a tuple x,y,c")
parser.add_argument(
    '-b', "--batch",
    help="Set the batch size", default=8, type=int)
parser.add_argument(
    '-e', "--epochs",
    help="Set the number of Epochs", default=1000, type=int)
parser.add_argument(
    '--mixed-precision',
    help="Use Mixed Precision. WARNING: May cause memory leaks", action="store_true")
parser.add_argument(
    '-f', '--lrfinder',
    help='Use the Learning Rate Finder on a model to determine the best learning rate range for said optimizer',
    choices=['adadelta', 'adagrad', 'adam', 'adamax', 'ftrl', 'nadam', 'rmsprop', 'sgd', 'sgd_nesterov'])
parser.add_argument(
    '--schedule',
    help="Set a Learning Rate Schedule, here either Polynomial Decay (polynomial) or Cyclic Learning Rate (cyclic) is Available",
    choices=['polynomial', 'cyclic'])
parser.add_argument(
    '--momentum',
    help='Only useful for lrfinder, adjusts momentum of an optimizer, if there is that option',
    type=float,
    default=0.0)
parser.add_argument(
    '-l', '--learning-rate',
    help='Set the learning rate')
schedule_arg = parser.add_argument_group(
    title='schedule', description='Arguments for the Learning Rate Scheduler')
schedule_arg.add_argument(
    '--max-lr', help='The maximum learning rate during training', type=float)
schedule_arg.add_argument(
    '--min-lr', help='The minimum learning rate during training', type=float)
schedule_arg.add_argument(
    '--power', help='The power used in Polynomial Decay', type=float)
schedule_arg.add_argument(
    '--cycle', help='The length of cycle used for Cyclic Learning Rates', type=int)

args = parser.parse_args()
# Get model_name
model_name = args.model
# Check the CITYSCAPES_ROOT path
if os.path.isdir(args.path):
    CITYSCAPES_ROOT = args.path
elif 'CITYSCAPES_DATASET' in os.environ:
    CITYSCAPES_ROOT = os.environ.get('CITYSCAPES_DATASET')
else:
    parser.error("ERROR: No valid path for Cityscapes Dataset given")
# Now do the target size
if args.target_size is None:
    target_size = (1024, 2048, 3)
else:
    target_size = tuple(int(i) for i in args.target_size.split(','))
# Get the Batch Size
batch_size = args.batch
# Get the number of epochs
epochs = args.epochs
# Check if the resume has been used, if so check the path
weights_path = args.resume
initial_epoch = 0
if args.resume is not None and os.path.isfile(args.resume):
    filename_split = os.path.basename(args.resume).split('-')
    initial_epoch = filename_split[1]
    time = filename_split[-1].replace('.hdf5', '')
elif args.resume is not None and not os.path.isfile(args.resume):
    parser.error("ERROR: Weights File not found.")
if args.momentum < 0:
    parser.error("ERROR: You specified a momentum value and it was below 0")
if args.schedule and (not args.max_lr or not args.min_lr):
    parser.error(
        "When using the scheduler, both the max and min learning rate must be set.")
if args.schedule == 'polynomial' and not args.power:
    parser.error("The power must be specified for Polynomal Decay")
if args.schedule == 'cyclic' and (not args.cycle or args.cycle < 0):
    parser.error(
        "Either cycle must be set when using Cyclic Learning Rate or you provided a cycle less than 0.")

# Initialize horovod
hvd.init()
gpus = tf.config.list_physical_devices('GPU')
# Set Memory Growth to alleviate memory issues
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
print("\nUsing {} GPUs via Horovod on rank {}\n".format(
    tf.config.get_visible_devices('GPU'),
    hvd.rank())
)

# Change policy to fp16 to use Tensor Cores
if args.mixed_precision:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)


class csMeanIoU(keras.metrics.Metric):
    """
    Very Similar to the keras Mean IoU 
    with some minor changes specific 
    to this workflow.
    """

    def __init__(self, num_classes, name=None, dtype=None):
        super(csMeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        self.cm = self.add_weight(
            'confusion_matrix',
            shape=(num_classes - 1, num_classes - 1),
            initializer=tf.zeros_initializer(),
            dtype=tf.float64,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Change y_true and y_pred to labels and flatten them
        y_true = tf.argmax(y_true, -1)
        y_pred = tf.argmax(y_pred, -1)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Get the Confusion Matrix
        curr_cm = tf.math.confusion_matrix(
            y_true, y_pred, self.num_classes, dtype=tf.float64)
        # Slice the Confusion Matrix to Ignore the Background
        curr_cm = tf.slice(
            curr_cm, (0, 0), (self.num_classes - 1, self.num_classes - 1))
        # Assign the weights of this to the cm of the class
        return self.cm.assign_add(curr_cm)

    def result(self):
        # Get True Positives, False Positives, False Negatives
        tp = tf.linalg.tensor_diag_part(self.cm)
        fp = tf.math.reduce_sum(self.cm, axis=0) - tp
        fn = tf.math.reduce_sum(self.cm, axis=1) - tp
        # Calculate the denominator
        denom = tp + fp + fn
        # Get the IoU of each class
        ious = tf.math.divide_no_nan(tp, denom)
        # Get the valid entries (Not 0)
        num_valid_entries = tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(denom, 0), dtype=tf.float64))
        # Return the Mean IoU of this image and Label
        return tf.math.divide_no_nan(tf.math.reduce_sum(ious), num_valid_entries)

    def reset_states(self):
        keras.backend.set_value(self.cm, np.zeros(
            (self.num_classes-1, self.num_classes-1)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(csMeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#----------CREATE DATASETS----------#
train_ds = datasets.create_dataset(
    CITYSCAPES_ROOT,
    batch_size, epochs,
    target_size,
    classes=CLASSES,
    shard_size=hvd.size(),
    shard_rank=hvd.rank()
)
val_ds = datasets.create_dataset(
    CITYSCAPES_ROOT,
    batch_size, epochs,
    target_size,
    train=False,
    classes=CLASSES,
    shard_size=hvd.size(),
    shard_rank=hvd.rank()
)
#----------CREATE MODEL AND BEGIN TRAINING----------#
time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
# Depending on model chosen, get the right model and create the optimizer for it
if model_name == 'unet':
    if args.learning_rate:
        learning_rate = args.learning_rate*hvd.size()
    else:
        learning_rate = 0.01*hvd.size()
    model = unet.model(input_size=target_size, num_classes=CLASSES)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
if model_name == 'separable_unet':
    if args.learning_rate:
        learning_rate = args.learning_rate*hvd.size()
    else:
        learning_rate = 0.003*hvd.size()
    model = separable_unet.model(input_size=target_size, num_classes=CLASSES)
    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
elif model_name == 'bayes_segnet':
    if args.learning_rate:
        learning_rate = args.learning_rate*hvd.size()
    else:
        learning_rate = 0.001*hvd.size()
    model = bayes_segnet.model(num_classes=CLASSES, input_size=target_size)
    optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=learning_rate)
elif model_name == 'fastscnn':
    if args.learning_rate:
        learning_rate = args.learning_rate*hvd.size()
    else:
        learning_rate = 0.045*hvd.size()
    model = fast_scnn.model(input_size=target_size, num_classes=CLASSES)
    optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=learning_rate)
elif model_name == 'deeplabv3+':
    if args.learning_rate:
        learning_rate = args.learning_rate*hvd.size()
    else:
        learning_rate = 0.007*hvd.size()
    model = deeplabv3plus.model(
        input_size=target_size, num_classes=CLASSES, depthwise=True, backbone='xception')
    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate)
# We are in LRFINDER mode
if args.lrfinder is not None:
    # Set a new optimizer for the LRFINDER
    if args.lrfinder == 'sgd_nesterov':
        optimizer = keras.optimizers.SGD(learning_rate=1e-6, nesterov=True)
    else:
        optimizer = keras.optimizers.get(args.lrfinder)
        optimizer.learning_rate = 1e-6
    if 'momentum' in optimizer.get_config():
        optimizer.momentum = args.momentum
    # Create distributed optimizer
    optimizer = hvd.DistributedOptimizer(optimizer)
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')],
        experimental_run_tf_function=False,
    )
    if hvd.rank() == 0:
        print("Finding Optimum Learning Rate Range")
    # Set the config for the lrfinder
    max_lr = 1e1
    epochs = 5
    num_steps = np.ceil(len(datasets.get_cityscapes_files(
        CITYSCAPES_ROOT, 'leftImg8bit', 'train', 'leftImg8bit')) / batch_size)
    num_steps = np.ceil(num_steps / hvd.size())
    # Create the learning rate finder
    lrf = lrfinder.LearningRateFinder(model)
    callbacks = []
    verbose = 0
    # Show the progress bar from only one worker
    if hvd.rank() == 0:
        verbose = 1
    # Use the callbacks for horovod necessary for metrics and variables
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
    ]
    # Start the Finder
    lrf.find(train_ds, 1e-6, max_lr, batch_size,
             num_steps, epochs, callbacks=callbacks, verbose=verbose)
    # Since we are using horovod, we only use the first worker to produce the plot
    if hvd.rank() == 0:
        # Create title for image, which will also be the name of the image file
        title = model_name + '-' + args.lrfinder
        if 'momentum' in optimizer.get_config():
            title = title + '-' + str(args.momentum)
        print("The Learning Rate Range has been found")
        # Create the path for the image file
        path = os.path.join('.', 'logs', model_name,
                            'lrfinder_results', title + '.png')
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        print("Plot the Loss")
        # Plot the loss
        lrf.plot_loss(title=title)
        # Save the figure
        plt.savefig(path, dpi=300, format='png')
        print("Please Look Here: {} for your results".format(path))

# Else we are training a model
else:
    optimizer = hvd.DistributedOptimizer(optimizer)
    # Whatever the model is, compile it
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            csMeanIoU(num_classes=CLASSES, name="meanIoU")
        ],
        experimental_run_tf_function=False
    )
    # Ensure directories are made for callbacks
    logs_dir = os.path.join('.', 'logs', model_name, time)
    csv_dir = os.path.join(logs_dir, 'csv')
    images_dir = os.path.join(logs_dir, 'images')
    if hvd.rank() == 0:
        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)
        if not os.path.isdir(csv_dir):
            os.mkdir(csv_dir)
        if not os.path.isdir(images_dir):
            os.mkdir(images_dir)

    # Custom Callback to save a prediction every epoch
    class prediction_on_epoch(keras.callbacks.Callback):
        def __init__(self, target_size):
            self.target_size = target_size

        def on_epoch_end(self, epoch, logs=None):
            image_to_predict = CITYSCAPES_ROOT + \
                '/leftImg8bit/val/frankfurt/frankfurt_000000_003025_leftImg8bit.png'
            # Get the byte strings for the files
            image_to_predict = tf.io.read_file(image_to_predict)
            # Decode the byte strings as png images
            image_to_predict = tf.image.decode_png(
                image_to_predict, channels=3)
            # Convert the image to float32 and scale all values between 0 and 1
            image_to_predict = tf.image.convert_image_dtype(
                image_to_predict, tf.float32)
            # Resize the image to target_size
            image_to_predict = tf.image.resize(
                image_to_predict, (self.target_size[0], self.target_size[1]), method='nearest')
            # Reshape it ready for prediction as a single batch!
            image_to_predict = tf.expand_dims(image_to_predict, 0)
            # Do a prediction
            prediction = model.predict(image_to_predict)
            # Reshape prediction
            prediction = tf.squeeze(prediction)
            # Now do matrix multiply
            image_to_save = np.matmul(prediction, CITYSCAPES_COLORS)
            # Save the Image
            Image.imsave(images_dir+'/epoch_{}.png'.format(epoch),
                         image_to_save.astype(np.uint8), dpi=300)

    class polyDecay(keras.callbacks.Callback):
        def __init__(self, min_lr, max_lr, steps, power, rank, warmup_epochs=3):
            self.max_lr = max_lr
            self.min_lr = min_lr
            self.steps = steps
            self.power = power
            self.rank = rank
            self.warmup_epochs = warmup_epochs

        def on_epoch_begin(self, epoch, logs=None):
            # If the optimizer has no learning_rate attribute we can't use the callback
            if not hasattr(self.model.optimizer, 'learning_rate'):
                raise ValueError(
                    'Optimizer must have a "learning_rate" attribute.')
            if epoch > self.warmup_epochs:
                # Get the previous learning rate
                previous = tf.identity(self.model.optimizer.learning_rate)
                # Work out the decay
                decay = (1 - epoch / self.steps) ** (self.power)
                # Change the current learning rate according to the decay and minimum and maximum learning rates
                self.model.optimizer.learning_rate = (
                    (self.max_lr - self.min_lr) * decay) + self.min_lr
                # If its the first worker, print the new learning rate
                if self.rank == 0:
                    print("Learning Rate has changed from {} to {}".format(
                        previous.numpy(), self.model.optimizer.learning_rate.numpy()))

    class cyclicLR(keras.callbacks.Callback):
        def __init__(self, min_lr, max_lr, cycle, rank, warmup_epochs=3):
            self.min_lr = min_lr
            self.max_lr = max_lr
            self.cycle = cycle
            self.step_size = (max_lr - min_lr)/cycle
            self.round_to = len(str(self.step_size))
            self.direction = 1
            self.rank = rank
            self.warmup_epochs = warmup_epochs

        def on_epoch_begin(self, epoch, logs=None):
            # If the optimizer has no learning_rate attribute we can't use the callback
            if not hasattr(self.model.optimizer, 'learning_rate'):
                raise ValueError(
                    'Optimizer must have a "learning_rate" attribute.')
            if epoch > self.warmup_epochs:
                # Get the previous learning rate
                previous = tf.identity(self.model.optimizer.learning_rate)
                # Get the new learning rate based on the direction we're going and the step size
                self.model.optimizer.learning_rate = np.round(
                    previous + self.step_size*self.direction, decimals=self.round_to)
                # Change direction based on whether we are going towards or away from the minimum or maximum learning rate
                if self.model.optimizer.learning_rate == self.min_lr or self.model.optimizer.learning_rate == self.max_lr:
                    self.direction *= -1
                # If its the first worker, print the new learning rate
                if self.rank == 0:
                    print("Learning Rate has changed from {} to {}".format(
                        previous.numpy(), self.model.optimizer.learning_rate.numpy()))

    # The callbacks  logs almost everything, and saves the best weights as the model trains, in theory last weights is best
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = x * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = x` ---> `lr = x * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
    ]
    # If we are using a Learning Rate Schedule then use the appropriate callbacks
    if args.schedule == 'polynomial':
        model.optimizer.learning_rate = args.max_lr*hvd.size()
        schedule = polyDecay(args.min_lr*hvd.size(), args.max_lr*hvd.size(),
                             epochs // hvd.size(), args.power, hvd.rank())
        callbacks.append(schedule)
    elif args.schedule == 'cyclic':
        model.optimizer.learning_rate = args.min_lr*hvd.size()
        schedule = cyclicLR(args.min_lr*hvd.size(),
                            args.max_lr*hvd.size(), args.cycle, hvd.rank())
        callbacks.append(schedule)
    # If its the chief worker, add the logging callbacks
    if hvd.rank() == 0:
        callbacks = callbacks + [
            keras.callbacks.CSVLogger(csv_dir+'/log.csv'),
            keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1),
            keras.callbacks.ModelCheckpoint(
                logs_dir+'/weights-{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}-'+time+'.hdf5', 'val_loss', mode='min', save_best_only=True),
            prediction_on_epoch(target_size)
        ]
    # Set the number of steps per epoch
    training_steps = np.ceil(len(datasets.get_cityscapes_files(
        CITYSCAPES_ROOT, 'leftImg8bit', 'train', 'leftImg8bit')) / batch_size)
    validation_steps = np.ceil(len(datasets.get_cityscapes_files(
        CITYSCAPES_ROOT, 'leftImg8bit', 'val', 'leftImg8bit')) / batch_size)
    verbose = 0
    if hvd.rank() == 0:
        verbose = 1
    training_steps = np.ceil(training_steps / hvd.size())
    validation_steps = np.ceil(validation_steps / hvd.size())
    # If we need to resume the training
    if args.resume is not None:
        model.load_weights(args.resume)

    # Train!
    history = model.fit(
        train_ds,
        epochs=epochs // hvd.size(),
        steps_per_epoch=training_steps,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
        initial_epoch=initial_epoch
    )
