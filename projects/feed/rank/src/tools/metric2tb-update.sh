pushd .
cd $1
port=$2
/home/gezi/mine/pikachu/tools/metrics2tb.py
`ps -xl | grep tensorboard | grep $port | awk '{print $3}' | /home/gezi/mine/pikachu/tools/kill-all.py`
CUDA_VISIBLE_DEVICES=-1 nohup tensorboard --logdir ./tb --port $port >> /tmp/metric2tb.$port.log 2>&1 &
popd
