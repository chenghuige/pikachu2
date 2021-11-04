#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   view.py
#        \author   chenghuige  
#          \date   2019-12-19 09:11:42.264984
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

  
from flask import render_template, Flask

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("render.html") 


