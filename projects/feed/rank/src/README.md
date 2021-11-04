# 运行前修改 conf/config.sh 里面的exp_root cloud_root 到自己的目标输出位置  

# 当前基线模型  
supervise.py train/v25/dlrm.sh (8尾号)
# 当前最佳模型 
supervise.py train/v25/dlrm-hm5.sh (+0.15%)  

# 按天训练 7天
sh ./scripts/video.sh --model_name=test --days=7 --end_hour=2019121522 --loop_type=day

# 按小时online训练 追齐到121722
sh ./scripts/video.sh --model_name=test -end_hour=2019121722 --loop_type=hour

# 关于按天小时训练的逻辑    
melt/apps/init.py  
melt/apps/train.py  --> fit  

# first type chgenv to set below see /root/.bashrc 
chenghuige/mine/pikachu/utils/:/search/odin/chenghuige/mine/pikachu:$PYTHONPATH;export LD_LIBRARY_PATH=/usr/local/cuda-10.0-cudnn-7.5/lib64:$LD_LIBRARY_PATH;'
alias chgenv='export PATH=/home/gezi/env/anaconda3/bin:/search/odin/chenghuige/tools/bin:/search/odin/chenghuige/tools/:/search/odin/chenghuige/soft/py3env/bin/:$PATH;export PYTHONPATH=/search/odin/chenghuige/mine/pikachu/utils/:/search/odin/chenghuige/mine/pikachu:$PYTHONPAH;export CUDA_HOME=/usr/local/cuda-10.0-cudnn-7.5/;export LD_LIBRARY_PATH=/home/gezi/env/anaconda3/lib/:/usr/local/cuda-10.0-cudnn-7.5/lib64:$LD_LIBRARY_PATH;. /home/gezi/env/anaconda3/etc/profile.d/conda.sh'
you can move to other machine be sure to set PATH to find anaconda3, LD PATH to find anaconda3 cuda10-cudnn7.5, PYTHONPATH to find pikachu

