from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import json
import traceback
import math
import copy
import collections
import hashlib
import os
reload(sys)
sys.setdefaultencoding('gbk')

for line in sys.stdin:
    line = line.rstrip().decode('gbk', 'ignore').encode('utf8', 'ignore')
    print(line)