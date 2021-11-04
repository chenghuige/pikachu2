'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 基于Pytorch多线程加载数据集
LastEditTime: 2019-09-17 12:59:37
'''



from importer import *
from config import *
from ..dataset import build_dataset,build_inferience_dataset
from ..transform import build_transforms,build_inference_transforms

def make_dataloader(cfg,is_train=True):
    '''
    Description: 
    Args (type): 
        cfg (yaml): config file.
        is_train (bool): True or False
    return: 
    '''
    DATALOADER = cfg.DATA.DATALOADER
    
    if is_train==True:
        shuffle = DATALOADER.TRAIN_SHUFFLE
        batch_size = DATALOADER.TRAIN_BATCH_SIZE
    else:
        shuffle = DATALOADER.VAL_SHUFFLE
        batch_size = DATALOADER.VAL_BATCH_SIZE
    num_workers = DATALOADER.NUM_WORKERS
    transforms = build_transforms(cfg,is_train)
    dataset = build_dataset(cfg,transforms=transforms,is_train=is_train)

    GLOBAL_SEED = 15
 
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    GLOBAL_WORKER_ID = None
    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)


    # dataloder = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=True,worker_init_fn=worker_init_fn)
    dataloder = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=True)
    return dataloder

def make_inference_dataloader(cfg):
    TOOLS = cfg.TOOLS
    image_n = TOOLS.image_n
    if image_n == 1:
        root_dir = TOOLS.root_dir_1_2
        csv_file = TOOLS.csv_file_1
    elif image_n == 2:
        root_dir = TOOLS.root_dir_1_2
        csv_file = TOOLS.csv_file_2
    elif image_n == 3:
        root_dir = TOOLS.root_dir_3_4
        csv_file = TOOLS.csv_file_3
    elif image_n == 4:
        root_dir = TOOLS.root_dir_3_4
        csv_file = TOOLS.csv_file_4
    elif image_n == 5:
        root_dir = TOOLS.root_dir_5_6
        csv_file = TOOLS.csv_file_5
    elif image_n == 6:
        root_dir = TOOLS.root_dir_5_6
        csv_file = TOOLS.csv_file_6
        
    shuffle = TOOLS.shuffle
    batch_size = TOOLS.batch_size
    num_workers = TOOLS.num_workers
    transforms = build_inference_transforms(cfg)
    dataset = build_inferience_dataset(root_dir=root_dir,csv_file=csv_file,transforms=transforms)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataloader
