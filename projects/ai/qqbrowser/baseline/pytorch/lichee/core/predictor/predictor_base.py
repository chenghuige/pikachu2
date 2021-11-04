import logging
import os
import shutil
import zipfile

from string import Template

import torch

from lichee import config
from lichee import plugin
from lichee.core import common
from lichee.utils import storage
from lichee.utils import sys_tmpfile
import json


@plugin.register_plugin(plugin.PluginType.PREDICTOR, "predictor_base")
class PredictorBase:
    def __init__(self, model_config_file):
        self.model_config_file = model_config_file
        config.merge_from_file(model_config_file)

        self.cfg = config.get_cfg()

        # gpu setting
        self.use_cuda = True
        self.master_gpu_id = 0
        self.gpu_ids = [0]
        self.init_gpu_setting()

        # init data model_loader
        self.eval_dataloader = None
        self.init_dataloader()

        # init model
        self.model = None
        self.init_model()

        # model inputs
        self.model_inputs = config.get_model_inputs()

        # init metrics
        self.metrics = []
        self.init_metrics()

        # init task cls
        self.task_cls = None
        self.init_task_cls()

        self.sample_inputs = None

        logging.info("predict config: %s", self.cfg)

    def init_gpu_setting(self):
        common.init_gpu_setting_default(self)

        if self.use_cuda:
            self.master_gpu_id = self.gpu_ids[0]

    def init_dataloader(self):
        common.init_dataloader_default(self)

    def init_model(self):
        model_cls = plugin.get_plugin(plugin.PluginType.UTILS_MODEL_LOADER, self.cfg.RUNTIME.EXPORT.TYPE)
        model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, self.cfg.RUNTIME.EXPORT.NAME)
        model_path = storage.get_storage_file(model_path)
        self.model = model_cls(model_path)
        logging.info("Start cpu training/evaluating...")

    def init_metrics(self):
        for metric_name in self.cfg.RUNTIME.METRICS.split(","):
            if metric_name == "none":
                continue
            metrics = plugin.get_plugin(plugin.PluginType.MODULE_METRICS, metric_name)
            self.metrics.append(metrics())

    def init_task_cls(self):
        self.task_cls = plugin.get_plugin(plugin.PluginType.TASK, self.cfg.MODEL.TASK.NAME)

    def predict(self):
        # self.eval_dataloader.dataset.is_training = False

        logging.info("predict start")
        is_write_head = False

        tmp_predict_res = sys_tmpfile.get_temp_file_path_once()
        f = open(tmp_predict_res, "w", encoding="utf-8")
        label_keys = []
        for step, batch in enumerate(self.eval_dataloader):
            inputs = self.get_inputs_batch(batch)
            label_keys, labels = self.get_label_batch(batch)
            label_vals = []
            with torch.no_grad():
                logits = self.model(inputs)

                model_outputs = self.task_cls.get_output(logits)
                if len(label_keys) != 0:
                    for label in labels:
                        label_vals.append(label.cpu().numpy())

            # get heads
            heads = self.get_result_heads(label_keys)

            # write heads once
            if not is_write_head:
                f.write("\t".join(heads) + "\n")
                is_write_head = True

            # get records
            record_arr = self.get_result_records(batch, label_vals, label_keys, model_outputs)

            if len(label_keys) > 0:
                for metric in self.metrics:
                    metric.collect(labels, logits)

            # write records
            for record in record_arr:
                f.write("\t".join(record) + "\n")

            if self.cfg.RUNTIME.DEBUG and (step + 1) > 4:
                break
        f.close()

        # upload predict result
        if "PREDICT" not in self.cfg.RUNTIME or self.cfg.RUNTIME.PREDICT.EXPORT_RESULT_PATH is None or \
                len(self.cfg.RUNTIME.PREDICT.EXPORT_RESULT_PATH) == 0:
            predict_res_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, 'predict_res.txt')
        else:
            predict_res_path = self.cfg.RUNTIME.PREDICT.EXPORT_RESULT_PATH
        # storage
        storage.put_storage_file(tmp_predict_res, predict_res_path)

        if len(label_keys) > 0:
            for metric in self.metrics:
                metric.calc()

        # if len(label_keys) != 0:
        #     logging.info("predict success. wrong prop: %s/%s", predict_wrong_num, predict_num)

        # export serving
        if "PREDICT" not in self.cfg.RUNTIME or self.cfg.RUNTIME.PREDICT.SHOULD_EXPORT_MODEL:
            logging.info("export serving start")
            self.export_serving()
            logging.info("export serving success")

    def export_serving(self):
        # read origin template
        template_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                    '../../template/tmlserving', self.cfg.RUNTIME.EXPORT.TYPE)
        ormbfile_path = os.path.join(template_dir, 'ormbfile.txt')
        ormbfile_string = open(ormbfile_path, 'r', encoding='utf-8').read()
        transformer_path = os.path.join(template_dir, 'transformer.txt')
        transformer_string = open(transformer_path, 'r', encoding='utf-8').read()

        # replace template
        t = Template(ormbfile_string)
        ormbfile_string = t.substitute(format=self.cfg.RUNTIME.EXPORT.TYPE, filename=self.cfg.RUNTIME.EXPORT.NAME)
        t = Template(transformer_string)
        transformer_string = t.substitute(output_key="output")

        # prepare tmp model dir
        tmp_dir = sys_tmpfile.get_temp_dir_once()
        tmp_ormbfile = os.path.join(tmp_dir, 'ormbfile.yaml')
        open(tmp_ormbfile, 'w', encoding='utf-8').write(ormbfile_string)
        tmp_transformer = os.path.join(tmp_dir, 'transformer.py')
        open(tmp_transformer, 'w', encoding='utf-8').write(transformer_string)
        tmp_config = os.path.join(tmp_dir, 'task.yaml')
        shutil.copy(self.model_config_file, tmp_config)
        tmp_model_dir = os.path.join(tmp_dir, 'model')
        tmp_model_path = os.path.join(tmp_model_dir, self.cfg.RUNTIME.EXPORT.NAME)
        os.makedirs(tmp_model_dir)
        model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, self.cfg.RUNTIME.EXPORT.NAME)
        local_model_path = storage.get_storage_file(model_path)
        shutil.copy(local_model_path, tmp_model_path)

        if self.cfg.RUNTIME.COMPRESS:
            tmp_zip = sys_tmpfile.get_temp_file_path_once()
            with zipfile.ZipFile(tmp_zip, "w") as write:
                write.write(tmp_ormbfile, arcname='tmp_export/ormbfile.yaml')
                write.write(tmp_transformer, arcname='tmp_export/transformer.py')
                write.write(tmp_config, arcname='tmp_export/task.yaml')
                write.write(tmp_model_path, arcname='tmp_export/model/' + self.cfg.RUNTIME.EXPORT.NAME)
            export_model_zip = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, 'tml_export.zip')
            storage.put_storage_file(tmp_zip, export_model_zip)
        else:
            # export without compress
            export_ormbfile_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, 'tml_export', 'ormbfile.yaml')
            storage.put_storage_file(tmp_ormbfile, export_ormbfile_path)
            export_transformer_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, 'tml_export', 'transformer.py')
            storage.put_storage_file(tmp_transformer, export_transformer_path)
            export_task_config_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, 'tml_export', 'task.yaml')
            storage.put_storage_file(tmp_config, export_task_config_path)
            export_model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, 'tml_export/model',
                                             self.cfg.RUNTIME.EXPORT.NAME)
            storage.put_storage_file(tmp_model_path, export_model_path)

    def get_result_heads(self, label_keys: list):
        heads = []
        for key in self.model_inputs:
            heads.append(key)
        heads.append("prediction")
        if len(label_keys) != 0:
            heads.extend(label_keys)
        return heads

    def get_result_records(self, batch, label_vals, label_keys, model_outputs):
        record_arr = []
        for i in range(model_outputs[0].shape[0]):
            record = []
            for key in self.model_inputs:
                if "org_" + key in batch:
                    record.append(str(batch["org_" + key][i]))
                elif key in batch:
                    record.append(str(batch[key][i]))

            outputs = []
            for item in model_outputs:
                outputs.append(item[i].tolist())
            record.append(json.dumps(tuple(outputs)))
            if len(label_keys) != 0:
                for idx in range(len(label_keys)):
                    record.append(str(label_vals[idx][i]))
            record_arr.append(record)
        return record_arr

    def get_inputs_batch(self, batch):
        return common.get_inputs_batch_default(self, batch)

    def get_label_batch(self, batch):
        return common.get_label_batch_default(self, batch)
