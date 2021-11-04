from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags
FLAGS = flags.FLAGS

import sys 
import os
import gezi
import mmh3
import six

ENCODING = 'gb18030'

buckets = 20000
if len(sys.argv) > 1:
  buckets = int(sys.argv[1]) 

field_ids = set()
for i, line in enumerate(sys.stdin):
  field_name = line.strip().split()[0]
  if six.PY2:
    field_id = mmh3.hash64(field_name.decode('utf8').encode(ENCODING))[0]
  else:
    field_id = mmh3.hash64(field_name.encode(ENCODING))[0]
    #field_id = gezi.hash_int64(field_name.encode(ENCODING))
  assert field_id != 0
  if field_id in field_ids:
    print('duplicate hash id', field_id, file=sys.stderr)
    exit(0)
  field_ids.add(field_id)
  print(field_name, field_id)

