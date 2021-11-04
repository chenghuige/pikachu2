# -*- coding: utf-8 -*-
import logging

import requests

from lichee import plugin
from lichee.utils import sys_tmpfile
from . import storage_base
from lichee import config


@plugin.register_plugin(plugin.PluginType.UTILS_STORAGE, "http")
class HttpStorage(storage_base.BaseStorage):
    """
    http storage
    """
    @classmethod
    def get_file(cls, url: str):
        if not url.startswith("http://"):
            url = "http://" + url
        return download_file(url)

    @classmethod
    def put_file(cls, src_file_path: str, dst_file_path: str):
        raise Exception("http does not support put file")


@plugin.register_plugin(plugin.PluginType.UTILS_STORAGE, "https")
class HttpsStorage(storage_base.BaseStorage):
    """
    https storage
    """
    @classmethod
    def get_file(cls, url: str):
        if not url.startswith("https://"):
            url = "https://" + url
        return download_file(url)

    @classmethod
    def put_file(cls, src_file_path: str, dst_file_path: str):
        raise Exception("https does not support put file")


def download_file(url):
    """
    download http file
    """
    tmp_file_path = sys_tmpfile.get_temp_file_path_once()

    logging.info("downloading %s to %s", url, tmp_file_path)

    sess = requests.Session()
    proxies = {}
    cfg = config.get_cfg()
    if "HTTP_PROXY" in cfg.RUNTIME.CONFIG:
        proxies["http"] = cfg.RUNTIME.CONFIG.HTTP_PROXY
    if "HTTPS_PROXY" in cfg.RUNTIME.CONFIG:
        proxies["https"] = cfg.RUNTIME.CONFIG.HTTPS_PROXY
    sess.proxies.update(proxies)
    with sess.get(url, stream=True) as r:
        r.raise_for_status()
        with open(tmp_file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return tmp_file_path
