source /root/.bashrc
chgenv
export PATH=/home/gezi/env/anaconda3/bin:/home/gezi/mine/pikachu/tools:/home/gezi/mine/pikachu/tools/bin:/home/gezi/soft/py3env/bin/:/usr/local/bin:/usr/bin:$PATH
export PYTHONPATH=/home/gezi/mine/pikachu/utils/:/home/gezi/mine/pikachu:$PYTHONPAH

start_hour=`date -d "-2 hours" "+%Y%m%d%H"`

sh ./loop-all.sh video $start_hour 3 3 &
sh ./loop-all.sh tuwen $start_hour 3 3 &

