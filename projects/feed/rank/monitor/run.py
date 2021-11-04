#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   run.py
#        \author   chenghuige  
#          \date   2019-12-19 09:11:37.991893
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from projects.common.monitor.view import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5001')


