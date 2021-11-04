# -*- coding: utf-8 -*-
import logging
import os

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

from lichee import config
from lichee import plugin
from lichee.utils import sys_tmpfile
from . import storage_base


@plugin.register_plugin(plugin.PluginType.UTILS_STORAGE, "cos")
class CosStorage(storage_base.BaseStorage):
    @classmethod
    def get_file(cls, file_path: str):
        (path, filename) = os.path.split(file_path)
        tmp_file_dir = sys_tmpfile.get_global_temp_dir()
        tmp_file_path = os.path.join(tmp_file_dir, filename)

        if os.path.exists(tmp_file_path):
            return tmp_file_path

        cfg = config.get_cfg()
        manager = CosManager(bucket=cfg.RUNTIME.CONFIG.COS.BUCKET,
                             secret_id=cfg.RUNTIME.CONFIG.COS.SECRET_ID,
                             secret_key=cfg.RUNTIME.CONFIG.COS.SECRET_KEY,
                             region=cfg.RUNTIME.CONFIG.COS.REGION,
                             token=None,
                             scheme="http",
                             with_agent=cfg.RUNTIME.CONFIG.COS.WITH_AGENT)
        logging.info("[cos] download %s to %s start", file_path, tmp_file_path)
        manager.cos_download(tmp_file_path, file_path)
        logging.info("[cos] download %s to %s success", file_path, tmp_file_path)
        return tmp_file_path

    @classmethod
    def put_file(cls, src_file_path: str, dst_file_path: str):
        if not super(CosStorage, cls).should_put_file():
            return

        cfg = config.get_cfg()
        manager = CosManager(bucket=cfg.RUNTIME.CONFIG.COS.BUCKET,
                             secret_id=cfg.RUNTIME.CONFIG.COS.SECRET_ID,
                             secret_key=cfg.RUNTIME.CONFIG.COS.SECRET_KEY,
                             region=cfg.RUNTIME.CONFIG.COS.REGION,
                             token=None,
                             scheme="http",
                             with_agent=cfg.RUNTIME.CONFIG.COS.WITH_AGENT)

        logging.info("[cos] upload %s to %s start", src_file_path, dst_file_path)
        manager.cos_upload(src_file_path, dst_file_path)
        logging.info("[cos] upload %s to %s success", src_file_path, dst_file_path)


class CosManager(object):
    def __init__(self, bucket="", secret_id="", secret_key="", region="", token=None, scheme="http", with_agent=True):
        """
        cos类【基于request上传下载文件，同步方式】

        :param bucket: cos的bucket
        :param secret_id: 用户的secretId
        :param secret_key: 用户的secretKey
        :param region: 用户的region
        :param token: 使用临时密钥需要传入 Token，默认为空，可不填
        :param scheme: 指定使用 http/https 协议来访问 cos，默认为 https，可不填
        """
        self.bucket = bucket
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.token = token
        self.scheme = scheme
        self.config = CosConfig(Region=self.region, SecretId=self.secret_id, SecretKey=self.secret_key,
                                Token=self.token, Scheme=self.scheme)

        # 2. 获取客户端对象
        self.client = CosS3Client(self.config)

    def cos_upload(self, local_filename, cos_filename):
        """
        上传文件

        :param local_filename: 本地文件地址
        :param cos_filename: cos地址
        :return response: None->异常, 非None->成功
        """
        with open(local_filename, 'rb') as fp:
            response = self.client.put_object(
                Bucket=self.bucket,
                Body=fp,
                Key=cos_filename,
                StorageClass='STANDARD',
                EnableMD5=False
            )
            return response

    def cos_download(self, local_filename, cos_filename):
        """
        下载文件

        :param local_filename: 本地文件地址
        :param cos_filename: cos地址
        :return response: None->异常, 非None->成功
        """
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=cos_filename,
        )
        response['Body'].get_stream_to_file(local_filename)
        return response
