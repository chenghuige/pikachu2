source /root/.bashrc
chgenv
export PATH=/home/gezi/env/anaconda3/bin:/home/gezi/mine/pikachu/tools:/home/gezi/mine/pikachu/tools/bin:/home/gezi/soft/py3env/bin/:/usr/local/bin:/usr/bin:$PATH
export PYTHONPATH=/home/gezi/mine/pikachu/utils/:/home/gezi/mine/pikachu:$PYTHONPAH

CUDA_VISIBLE_DEVICES=-1 python ../tools/eval-days.py /home/gezi/tmp/rank/data/video_hour_sgsapp_v1/exps/monitor --parallel=day &
CUDA_VISIBLE_DEVICES=-1 python ../tools/eval-days.py /home/gezi/tmp/rank/data/tuwen_hour_sgsapp_v1/exps/monitor --parallel=day &

