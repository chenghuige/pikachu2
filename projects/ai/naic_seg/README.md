1. 安装依赖  
pip install -r ./requirements.txt  
2. 设置PYTHONPATH
如在colab设置如下 取决放置pikachu的位置  
sys.path.append('/content/pikachu/utils')  
sys.path.append('/content/pikachu/third')  
sys.path.append('/content/pikachu')  
sys.path.append('/content/pikachu/projects/ai/naic_seg')  
3. 进入运行目录 
cd /userhome/pikachu/projects/ai/naic_seg 
sh run.sh   
如果已经生成tfrecord 可以直接 
cd /userhome/pikachu/projects/ai/naic_seg/gseg  
sh run.sh  
4. 配置主要需要配置可以在 
/userhome/pikachu/projects/ai/naic_seg/gseg/scripts/run-quarter.sh  
设置 
--batch_szie=32 # 不OOM的情况下 按照 8 16 32 64 可以考虑设置最大 colab tpu环境采用 32  
--num_gpus=4  # 可以设置系统最大gpu数目，colab tpu环境自动按照环境tpu核心数目设置为8了  
5. 生成的submit.zip路径  
/userhome/pikachu/projects/ai/naic_seg/gseg/infer/submit.zip 


