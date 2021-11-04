import sys
import time
import requests
import traceback
import json
from tqdm import tqdm

##
# @file KV.py
# @Synopsis KV类，KV python client
# @author cuice
# @version 1.3

# 如果需要获取结果为str(gbk/utf8)，将r.text替换为r.content即可
class KV(object):
    def __init__(self, appid='110216', namespace='article_forward_index',
                 pool_connections=10, pool_maxsize=100, max_retries=3):
        self.url_base = "http://kv.sogou/"
        self.url = "/".join([self.url_base, appid, namespace, ""])
        self.murl = "/".join([self.url_base, "mget", appid, namespace, ""])
        self.session = self.get_http_session(pool_connections, pool_maxsize, max_retries)
        self.pool_size = pool_maxsize
        

    def get_http_session(self, pool_connections, pool_maxsize, max_retries):
        session = requests.Session()
        # 创建一个适配器，连接池的数量pool_connections, 最大数量pool_maxsize, 失败重试的次数max_retries
        adapter = requests.adapters.HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize, max_retries=max_retries)
        # 告诉requests，http协议和https协议都使用这个适配器
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def get(self, key):
        url = self.url + key
        try:
            r = self.session.get(url, timeout=1)
            if r.status_code == 200:
                r.encoding = "gbk"
                #return r.text # unicode format
                return r.content
            elif r.status_code == 404:
                return None
            else:
                print("Response from kv server error %s when trying to get..." %str(r.status_code), file=sys.stderr)
                print("response:%s" % r.text, file=sys.stderr)
                print("url:%s" % url, file=sys.stderr)
                return None
        except Exception as e:
            print("Get record from kv error!", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return None

    def set(self, key, value, expire_time=-1):
        if expire_time >= 0:
            url = self.url + key + "?expire=" + str(expire_time)
        else:
            url = self.url + key
        try:
            if isinstance(value, str):
                value = value.encode("gbk")
            elif not isinstance(value, bytes):
                value = str(value)
            r = self.session.post(url, data=value, timeout=1)
            if r.status_code == 200:
                return True
            else:
                print("Response from kv server error %s when trying to set..." %str(r.status_code), file=sys.stderr)
                print("response : %s" % r.text, file=sys.stderr)
                print("url : %s" % url, file=sys.stderr)
                return False
        except Exception as e:
            print("Set record to kv error!", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return False

    def remove(self, key):
        url = self.url + key
        try:
            r = self.session.delete(url, timeout=1)
            if r.status_code == 200:
                return True
            elif r.status_code == 404:
                return True
            else:
                print("Response from kv server error %s when trying to remove..." %str(r.status_code), file=sys.stderr)
                print(r.text, file=sys.stderr)
                return False
        except Exception as e:
            print("Remove record from kv error!", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return False

    def mget(self, key_list):
        url = self.murl
        value = "\n".join(key_list)
        try:
            r = self.session.post(url, data=value, timeout=1)
            if r.status_code == 200:
                r.encoding = "gbk"
                res = self.analysis_mget_res(r.text)
                if len(res) != len(key_list):
                    raise ValueError("Lengthes of request and response are inconsistent.")
                return res
            else:
                # print("Response from kv server error %s when trying to mget..." %str(r.status_code), file=sys.stderr)
                # print(r.text, file=sys.stderr)
                return None
        except Exception as e:
            # print("Mget records from kv error!", file=sys.stderr)
            # traceback.print_exc(file=sys.stderr)
            return None
        
    # 需要保证查询的value不以"$"开头
    def analysis_mget_res(self, str):
        res = []
        s = str.split("\r\n")
        for item in s[1:-1]:
            if not item.startswith("$"):
                res.append(item)
            elif item[1:] == "-1":
                res.append("")
        return res
    
    def get_expiration(self, key):
        url = self.url + key
        try:
            r = self.session.get(url, timeout=1)
            if r.status_code == 200:
                header = r.headers
                if "Expiration-Time" not in header:
                    return -1
                GMT_FORMAT = '%a, %d %b %Y %H:%M:%S GMT'
                expire_time = time.mktime(time.strptime(header['Expiration-Time'], GMT_FORMAT))+8*3600
                return expire_time
            elif r.status_code == 404:
                return None
            else:
                print("Response from kv server error %s when trying to get..." %str(r.status_code), file=sys.stderr)
                print(r.text, file=sys.stderr)
                return None
        except Exception as e:
            print("Get record from kv error!", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return None

    def get_titles(self, docids, pool_size=100):
        res = {}
        num_steps = -(-len(docids) // pool_size)
        for i in tqdm(range(num_steps), ascii=True, desc='get_titles'):
            docids_ = docids[i * pool_size:(i+1) * pool_size]
            docs = self.mget(docids_)
            if not docs:
                # print(i, 'not get', file=sys.stderr)
                continue
            for j, did in enumerate(docids_):
                res[did] = json.loads(docs[j])['title']
        return res

    def get_infos(self, docids, key=None, pool_size=100):
        res = {}
        num_steps = -(-len(docids) // pool_size)
        for i in tqdm(range(num_steps), ascii=True, desc='get_infos'):
            docids_ = docids[i * pool_size:(i+1) * pool_size]
            docs = self.mget(docids_)
            if not docs:
                # print(i, 'not get', file=sys.stderr)
                continue
            for j, did in enumerate(docids_):
                if key:
                    if ',' not in key:
                       res[did] = json.loads(docs[j])[key]
                    else:
                        keys = key.split(',')
                        m = {}
                        info = json.loads(docs[j])
                        for key in keys:
                            m[key] = info[key]
                        res[did] = m
                else:
                    res[did] = json.loads(docs[j])
        return res
