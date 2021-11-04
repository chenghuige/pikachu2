#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   crawl.py
#        \author   chenghuige  
#          \date   2021-08-21 09:12:19.043108
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import gezi

import random
import functools
import time

import cv2
from PIL import Image
from io import BytesIO

import requests               
import json
from collections import OrderedDict, defaultdict, ChainMap
from lxml import etree

# python crawl.py 6299446167744660910
# python crawl.py 8181862433864400302
if __name__ == '__main__':      
    id = sys.argv[1]        
    url = f'https://kandianshare.html5.qq.com/v3/video/{id}'     
    ic(url)  
    req = requests.get(url=url)        
    ic(req.text)          
    html = req.text        
    dom = etree.HTML(html)
    ic(dom)
    hrefs = dom.xpath(u"//script[@id='config']")
    # ic([(i, x.text) for i, x in enumerate(hrefs)])
    assert len(hrefs) == 1
    text = hrefs[0].text[len('window._configs='):]
    m = json.loads(text)
    # ic(m)
    # ic(list(m.keys()))
    keys = ['vid', 'title', 'tags', 'imageUrl', 'mediaInfo', 'subject', 'type', 'src', 'totalTime']

    x = OrderedDict()
    m = m['videoContent']
    for key in keys:
      if key == 'mediaInfo':
        x['mediaName'] = m[key]['mediaName']
      elif key == 'tags':
        x['tags'] = ','.join([x['sTagName'] for x in m['tags']])
        x['tag_weights'] = ','.join([str(x['fWeight']) for x in m['tags']])
      else:
        x[key] = m[key]
    
    ic(x)
