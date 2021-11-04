import logging
import os
from pprint import pprint

import tensorflow as tf

from config import parser
from data_helper import create_datasets
from metrics import Recorder
from model import MultiModal
from util import test_spearmanr

import gezi
from gezi import tqdm

def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    # 2. build model
    model = MultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder(), Recorder()

    # 5. define train and valid step function
    @tf.function
    def train_step(inputs):
        labels = inputs['labels']
        with tf.GradientTape() as tape:
            predictions, _ = model(inputs, training=True)
            loss = loss_object(labels, predictions) * labels.shape[-1]  # convert mean back to sum
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, labels, predictions)

    @tf.function
    def val_step(inputs):
        vids = inputs['vid']
        labels = inputs['labels']
        predictions, embeddings = model(inputs, training=False)
        loss = loss_object(labels, predictions) * labels.shape[-1]  # convert mean back to sum
        val_recorder.record(loss, labels, predictions)
        return vids, embeddings

    # 6. training
    import gezi
    num_examples = gezi.read_int('./data/num_train.txt')
    ic(num_examples, args.batch_size)
    num_steps_per_epoch = -(-num_examples // args.batch_size)
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        for train_batch in tqdm(train_dataset, total=num_steps_per_epoch):
            checkpoint.step.assign_add(1)
            step = checkpoint.step.numpy()
            if step > args.total_steps:
                break
            train_step(train_batch)
            if step % args.print_freq == 0:
                train_recorder.log(epoch, step)
                train_recorder.reset()

            # 7. validation
            if step % args.eval_freq == 0:
                import gezi
                with gezi.Timer('eval'):
                    vid_embedding = {}
                    for val_batch in val_dataset:
                        vids, embeddings = val_step(val_batch)
                        for vid, embedding in zip(vids.numpy(), embeddings.numpy()):
                            vid = vid.decode('utf-8')
                            vid_embedding[vid] = embedding
                # 8. test spearman correlation
                spearmanr = test_spearmanr(vid_embedding, args.annotation_file)
                val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                val_recorder.reset()

                # 9. save checkpoints
                if spearmanr > 0.45:
                    checkpoint_manager.save(checkpoint_number=step)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = parser.parse_args()

    if not os.path.exists(args.savedmodel_path):
        os.mkdir(args.savedmodel_path)

    pprint(vars(args))
    train(args)


if __name__ == '__main__':
    main()
