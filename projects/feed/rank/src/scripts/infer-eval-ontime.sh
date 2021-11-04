source /root/.bashrc
chgenv
export PATH=/home/gezi/env/anaconda3/bin:/home/gezi/mine/pikachu/tools:/home/gezi/mine/pikachu/tools/bin:/home/gezi/soft/py3env/bin/:/usr/local/bin:/usr/bin:$PATH
export PYTHONPATH=/home/gezi/mine/pikachu/utils/:/home/gezi/mine/pikachu:$PYTHONPAH

start_hour=`date -d "-2 hours" "+%Y%m%d%H"`

sh ./infer-eval.sh video $start_hour 24 1 &
sh ./infer-eval.sh tuwen $start_hour 24 1 &

