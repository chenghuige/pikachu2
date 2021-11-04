#!/bin/python


import json
import sys

def parse_info(read_path,write_path,model_name,model_ver):
  with open(read_path,'r') as f:
    line = f.readline()
    json_info = json.loads(line)
    json_info['model_name'] = model_name
    json_info['model_ver'] = model_ver
    res = json.dumps(json_info)
    with open(write_path,'w') as f:
      f.write(res)

read_path = sys.argv[1]
write_path = sys.argv[2]
model_name = sys.argv[3]
model_ver = sys.argv[4]

parse_info(read_path,write_path,model_name,model_ver)
