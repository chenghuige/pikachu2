# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TQDM Progress Bar."""

import time
import tensorflow as tf
from collections import defaultdict
from typeguard import typechecked

from tensorflow.keras.callbacks import Callback
from absl import flags

FLAGS = flags.FLAGS

import gezi
import melt


# @tf.keras.utils.register_keras_serializable(package="Addons")
class TQDMProgressBar(Callback):
  """TQDM Progress Bar for Tensorflow Keras.

    Args:
        metrics_separator: Custom separator between metrics.
            Defaults to ' - '.
        overall_bar_format: Custom bar format for overall
            (outer) progress bar, see https://github.com/tqdm/tqdm#parameters
            for more detail.
        epoch_bar_format: Custom bar format for epoch
            (inner) progress bar, see https://github.com/tqdm/tqdm#parameters
            for more detail.
        update_per_second: Maximum number of updates in the epochs bar
            per second, this is to prevent small batches from slowing down
            training. Defaults to 10.
        metrics_format: Custom format for how metrics are formatted.
            See https://github.com/tqdm/tqdm#parameters for more detail.
        leave_epoch_progress: True to leave epoch progress bars.
        leave_overall_progress: True to leave overall progress bar.
        show_epoch_progress: False to hide epoch progress bars.
        show_overall_progress: False to hide overall progress bar.
    """

  @typechecked
  def __init__(
      self,
      desc: str = None,
      leave: bool = True,
      metrics_separator: str = " - ",
      overall_bar_format: str = "{l_bar}{bar} {n_fmt}/{total_fmt} ETA: "
      "{remaining}s, {rate_fmt}{postfix}",
      epoch_bar_format: str = "{n_fmt}/{total_fmt}{bar} ETA: "
      "{remaining}s - {desc}",
      metrics_format: str = "{value:0.4f}",
      update_per_second: int = 10,
      leave_epoch_progress: bool = True,
      leave_overall_progress: bool = True,
      show_epoch_progress: bool = True,
      show_overall_progress: bool = True,
      steps_per_execution: int = 1,
      update_each_epoch: bool = True,
      initial_epoch: int = 0,
  ):

    try:
      # import tqdm here because tqdm is not a required package
      # for addons
      import tqdm

      version_message = "Please update your TQDM version to >= 4.36.1, "
      "you have version {}. To update, run !pip install -U tqdm"
      assert tqdm.__version__ >= "4.36.1", version_message.format(
          tqdm.__version__)
      from tqdm.auto import tqdm

      self.tqdm = tqdm
    except ImportError:
      raise ImportError("Please install tqdm via pip install tqdm")

    self.metrics_separator = metrics_separator
    self.overall_bar_format = overall_bar_format
    self.epoch_bar_format = epoch_bar_format
    self.leave_epoch_progress = leave_epoch_progress
    self.leave_overall_progress = leave_overall_progress
    self.show_epoch_progress = show_epoch_progress
    self.show_overall_progress = show_overall_progress
    self.metrics_format = metrics_format

    # compute update interval (inverse of update per second)
    self.update_interval = 1 / update_per_second

    self.last_update_time = time.time()
    self.last_update_time2 = time.time()
    self.overall_progress_tqdm = None
    self.epoch_progress_tqdm = None
    self.is_training = False
    self.num_epochs = None
    self.logs = None
    self.steps_per_execution = steps_per_execution
    self.update_each_epoch = update_each_epoch
    self.epoch_progbar_inited = False
    self.initial_epoch = initial_epoch
    self.desc = desc

    if not leave:
      self.leave_epoch_progress = False
      self.leave_overall_progress = False
    super().__init__()

  def _initialize_progbar(self, hook, epoch, logs=None):
    self.num_samples_seen = 0
    self.steps_to_update = 0
    self.steps_so_far = 0
    self.logs = defaultdict(float)
    self.num_epochs = self.params["epochs"]
    # self.mode = "steps"
    self.mode = 'it'
    self.total_steps = self.params["steps"]
    # ic(self.num_epochs, self.total_steps)
    if not self.update_each_epoch:
      self.total_steps *= self.num_epochs
    
    if hook == "train_overall":
      if self.show_overall_progress:
        self.overall_progress_tqdm = self.tqdm(
            desc=self.desc or "Training",
            total=self.num_epochs,
            # bar_format=self.overall_bar_format,
            leave=self.leave_overall_progress,
            dynamic_ncols=True,
            initial=self.initial_epoch,
            unit="epochs",
            ascii=False)
    elif hook == "test":
      if self.show_epoch_progress:
        self.epoch_progress_tqdm = self.tqdm(
            total=self.total_steps,
            desc=self.desc or "Evaluating",
            # bar_format=self.epoch_bar_format,
            leave=self.leave_epoch_progress,
            dynamic_ncols=True,
            unit=self.mode,
            ascii=False)
    elif hook == "predict":
      if self.show_epoch_progress:
        self.epoch_progress_tqdm = self.tqdm(
            total=self.total_steps,
            desc=self.desc or "Predicting",
            # bar_format=self.epoch_bar_format,
            leave=self.leave_epoch_progress,
            dynamic_ncols=True,
            unit=self.mode,
            ascii=False)
    elif hook == "train_epoch":
      try:
        train_hour = FLAGS.train_hour if FLAGS.loop_train else None
        current_epoch_description = 'Epoch:%2d/%d' % (
            epoch + 1,
            sefl.num_epochs) if not train_hour else '%s-%d/%d Epoch: %2d/%d' % (
                train_hour, FLAGS.round + 1, FLAGS.num_rounds, epoch + 1,
                self.num_epochs)
      except Exception:
        current_epoch_description = "Epoch:%2d/%d" % (epoch + 1,
                                                      self.num_epochs)
      if not self.update_each_epoch:
        # current_epoch_description = f'Epochs:{self.initial_epoch + 1}-{self.num_epochs}' if self.initial_epoch else f'Epochs:{self.num_epochs}'
        current_epoch_description = f'Epochs:{self.num_epochs}'
      try:
        if FLAGS.model_name:
          current_epoch_description = f'[{FLAGS.model_name}] {current_epoch_description}'
      except Exception:
        pass
      if self.show_epoch_progress:
        # print(current_epoch_description)
        self.epoch_progress_tqdm = self.tqdm(
            total=self.total_steps,
            desc=current_epoch_description,
            # bar_format=self.epoch_bar_format,
            leave=self.leave_epoch_progress,
            dynamic_ncols=True,
            unit=self.mode,
            initial=0 if self.update_each_epoch else melt.get_total_step(),
            ascii=False,
        )

  def _clean_up_progbar(self, hook, logs):
    if hook == "train_overall":
      if self.overall_progress_tqdm:
        self.overall_progress_tqdm.close()
    else:
      if hook == "test" or hook == "predict":
        metrics = self.format_metrics(logs, self.num_samples_seen)
      else:
        metrics = self.format_metrics(logs)
      if self.show_epoch_progress:
        # self.epoch_progress_tqdm.desc = metrics
        self.epoch_progress_tqdm.set_postfix(metrics)
        # set miniters and mininterval to 0 so last update displays
        self.epoch_progress_tqdm.miniters = 0
        self.epoch_progress_tqdm.mininterval = 0
        # update the rest of the steps in epoch progress bar
        self.epoch_progress_tqdm.update(self.total_steps -
                                        self.epoch_progress_tqdm.n)
        self.epoch_progress_tqdm.close()

  def _update_progbar(self, logs):
    steps_per_execution = self.steps_per_execution
    if self.mode == "samples":
      batch_size = logs["size"]
    else:
      batch_size = 1
    # ic(steps_per_execution, self.num_samples_seen)
    self.num_samples_seen += batch_size * steps_per_execution
    # ic(self.num_samples_seen)
    self.steps_to_update += steps_per_execution
    self.steps_so_far += steps_per_execution
    # ic(self.steps_so_far)

    if self.steps_so_far <= self.total_steps:
      for metric, value in logs.items():
        ## TODO HACK IoU被禁止在训练过程中更新 因为当前无法做到一定train step做一次IoU计算以及 每次自动reset state 还有就是valid和train混合更新相同IoU问题
        ## melt.metrics使用的那么都是首字母大写
        # if metric[0].isupper() or (metric.startswith('val_') and metric[4].isupper()):
        if FLAGS.pb_loss_only and not 'loss' in metric:
          continue
        if self.is_training and metric[0].isupper():
          continue
        # self.logs[metric] += value * batch_size
        ## 改成只展现当前metric result
        self.logs[metric] = value

      now = time.time()
      time_diff = now - self.last_update_time
      if self.show_epoch_progress and time_diff >= self.update_interval:

        # update the epoch progress bar
        metrics = self.format_metrics(self.logs, self.num_samples_seen)
        # self.epoch_progress_tqdm.desc = metrics
        self.epoch_progress_tqdm.set_postfix(metrics)
        self.epoch_progress_tqdm.update(self.steps_to_update)

        # reset steps to update
        self.steps_to_update = 0

        # update timestamp for last update
        self.last_update_time = now

    if self.overall_progress_tqdm and FLAGS.eval_verbose == 0 and FLAGS.show_eval:
      metrics = gezi.get('Metrics', {})
      now = time.time()
      time_diff = now - self.last_update_time2
      if metrics and time_diff >= self.update_interval:
        self.overall_progress_tqdm.set_postfix(metrics)
        self.last_update_time2 = now

  def on_train_begin(self, logs=None):
    self.is_training = True
    self._initialize_progbar("train_overall", None, logs)

  def on_train_end(self, logs={}):
    self.is_training = False
    self._clean_up_progbar("train_overall", logs)

  def on_test_begin(self, logs={}):
    if not self.is_training:
      self._initialize_progbar("test", None, logs)

  def on_test_end(self, logs={}):
    if not self.is_training:
      self._clean_up_progbar("test", self.logs)

  def on_predict_begin(self, logs={}):
    if not self.is_training:
      self._initialize_progbar("predict", None, logs)

  def on_predict_end(self, logs={}):
    if not self.is_training:
      self._clean_up_progbar("predict", self.logs)

  def on_epoch_begin(self, epoch, logs={}):
    if self.update_each_epoch or not self.epoch_progbar_inited:
      self._initialize_progbar("train_epoch", epoch, logs)
      self.epoch_progbar_inited = True

  def on_epoch_end(self, epoch, logs={}):
    if self.update_each_epoch:
      self._clean_up_progbar("train_epoch", logs)
    if self.overall_progress_tqdm and FLAGS.eval_verbose == 0 and FLAGS.show_eval:
      metrics = gezi.get('Metrics', {})
      if metrics:
        self.overall_progress_tqdm.set_postfix(metrics)
      self.overall_progress_tqdm.update(1)

    # if FLAGS.num_epochs2 and (epoch + 1) != FLAGS.num_epochs and (epoch + 1) % FLAGS.num_epochs2 == 0:
    #     exit(0)

  def on_test_batch_end(self, batch, logs={}):
    if not self.is_training:
      self._update_progbar(logs)

  def on_predict_batch_end(self, batch, logs={}):
    # TODO here logs..
    logs = {}
    if not self.is_training:
      self._update_progbar(logs)

  def on_batch_end(self, batch, logs={}):
    self._update_progbar(logs)

  def format_metrics(self, logs={}, factor=1):
    """Format metrics in logs into a string.

        Arguments:
            logs: dictionary of metrics and their values. Defaults to
                empty dictionary.
            factor (int): The factor we want to divide the metrics in logs
                by, useful when we are computing the logs after each batch.
                Defaults to 1.

        Returns:
            metrics_string: a string displaying metrics using the given
            formators passed in through the constructor.
        """

    metrics = {}
    for key, value in logs.items():
      if key in ["batch", "size"]:
        continue
      # value = self.metrics_format.format(value=value / factor)
      value = self.metrics_format.format(value=value)
      metrics[key] = value
    return metrics

  def get_config(self):
    config = {
        "metrics_separator": self.metrics_separator,
        "overall_bar_format": self.overall_bar_format,
        "epoch_bar_format": self.epoch_bar_format,
        "leave_epoch_progress": self.leave_epoch_progress,
        "leave_overall_progress": self.leave_overall_progress,
        "show_epoch_progress": self.show_epoch_progress,
        "show_overall_progress": self.show_overall_progress,
    }

    base_config = super().get_config()
    return {**base_config, **config}
