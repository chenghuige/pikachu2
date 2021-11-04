#!/bin/python
import sys

read_file = sys.argv[1]  # input file path
item_name = sys.argv[2]

metrics = {}
with open(read_file, 'r') as f:
    for line in f.readlines()[::-1]:
        if "valid_metrics:" in line:
            line = line[line.index("valid_metrics:") + len("valid_metrics:"):]
            items = eval(line)
            for item in items:
                name, value = item.split(":")
                metrics[name] = value
            break
if item_name == "all":
    result=[]
    for key,value in metrics.items():
        result.append("%s:%s"%(key,value))
    print(" ".join(result))
elif item_name == "important":
    keys = ['version', 'click_ratio', \
            'time_per_user', 
            # 'time_per_show', \
            'time_per_click', \
            'gold/auc', \
            'group/auc', \
            'group/click/time_auc', \
            # 'group/top1_best', 
            'group/top3_best', \
            'auc', 'time_auc', 'weighted_time_auc', \
            'click/time_auc', 'click/weighted_time_auc', \
            'group/time_auc', 'group/weighted_time_auc', 'group/top3_click_rate', \
            'group/click/weighted_time_auc', 'group/click/top3_rate'
           ]
    def rename(key):
        return key.replace('weighted_time', 'wtime') \
                  .replace('version', 'v') \
                  .replace('group', 'g') \
                  .replace('/', '_')
    result=[]
    for key in keys:
        if key in metrics:
            value = metrics[key]
            # try:
            #     value = "%.4f" % float(value)
            # except Exception:
            #     pass
            result.append("%s:%s"%(rename(key), value))
    print("\n".join(result))
elif item_name in metrics:
    print(metrics[item_name])
else:
    print("None")
