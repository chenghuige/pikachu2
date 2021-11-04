import logging
import tensorflow as tf


class Recorder:
    def __init__(self):
        self.loss = tf.keras.metrics.Mean()
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

        self.pattern = 'Epoch: {}, step: {}, loss: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'

    def record(self, losses, labels, predictions):
        self.loss.update_state(losses)
        self.precision.update_state(labels, predictions)
        self.recall.update_state(labels, predictions)

    def reset(self):
        self.loss.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()

    def _results(self):
        loss = self.loss.result().numpy()
        precision = self.precision.result().numpy()
        recall = self.recall.result().numpy()
        f1 = 2 * precision * recall / (precision + recall)
        return [loss, precision, recall, f1]

    def score(self):
        return self._results()[-1].numpy()

    def log(self, epoch, num_step, prefix='', suffix=''):
        loss, precision, recall, f1 = self._results()
        logging.info(prefix + self.pattern.format(epoch, num_step, loss, precision, recall, f1) + suffix)
