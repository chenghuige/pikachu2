'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
@LastEditTime: 2019-09-02 16:44:20
'''
from argparse import ArgumentParser
from os import mkdir,path
import sys
sys.path.append('..')
# import os
# print(os.getcwd())
from config import cfg
from config import *
from data.dataloader import make_dataloader,make_inference_dataloader
from engine import do_train
from model import build_model
from solver import make_optimizer
from solver import make_criterion
from solver import make_lr_scheduler
from util import make_metrics
from importer import *


def train(cfg):
    model = build_model(cfg)
    # model.fix_bn()
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    optimizer = make_optimizer(cfg,model)
    criterion = make_criterion(cfg)
    scheduler = make_lr_scheduler(cfg,optimizer)
    metrics = make_metrics(cfg)
    train_loader = make_dataloader(cfg,is_train=True)
    val_loader = make_dataloader(cfg,is_train=False)

    cfg.TOOLS.image_n = 3
    #image_3_dataloader = make_inference_dataloader(cfg=cfg)
    image_3_dataloader = None
    cfg.TOOLS.image_n = 4
    #image_4_dataloader = make_inference_dataloader(cfg=cfg)
    image_4_dataloader = None

    do_train(cfg,model=model,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,
                    scheduler=scheduler,loss_fn=criterion,metrics=metrics,image_3_dataloader=image_3_dataloader,image_4_dataloader=image_4_dataloader)



def load_arg():
    parser = ArgumentParser(description="Pytorch Hand Detection Training")
    parser.add_argument("-config_file","--CONFIG_FILE",type=str,help="Path to config file")
    parser.add_argument('-log_period',"--LOG_PERIOD",type=float,help="Period to log info")
    parser.add_argument("-val_period","--VAL_PERIOD",type=float)
    parser.add_argument("-tag","--TAG",type=str)
    
    # DATA
    parser.add_argument("-num_workers", "--DATA.DATALOADER.NUM_WORKERS",type=int,
                        help='Num of data loading threads. ')
    parser.add_argument("-train_batch_size","--DATA.DATALOADER.TRAIN_BATCH_SIZE",type=int,
                        help="input batch size for training (default:64)")
    parser.add_argument("-val_batch_size","--DATA.DATALOADER.VAL_BATCH_SIZE",type=int,
                        help="input batch size for validation (default:128)")
    parser.add_argument("-train_csv_file","--DATA.DATASET.train_csv_file",type=str)
    parser.add_argument("-train_root_dir","--DATA.DATASET.train_root_dir",type=str)
    parser.add_argument("-train_mask_dir","--DATA.DATASET.train_mask_dir",type=str)
    parser.add_argument("-val_csv_file","--DATA.DATASET.val_csv_file",type=str)
    parser.add_argument("-val_root_dir","--DATA.DATASET.val_root_dir",type=str)
    parser.add_argument("-val_mask_dir","--DATA.DATASET.val_mask_dir",type=str)

    # MODEL
    parser.add_argument('-model',"--MODEL.NET_NAME",type=str,
                        help="Net to build")
    parser.add_argument('-path',"--MODEL.LOAD_PATH",type=str,
                        help="path/file of a pretrain model(state_dict)")
    parser.add_argument("-device","--MODEL.DEVICE",type=str,
                        help="cuda:x (default:cuda:0)")

    # SOLVER
    parser.add_argument("-max_epochs","--SOLVER.MAX_EPOCHS",type=int,
                        help="num of epochs to train (default:50)")
    parser.add_argument('-optimizer',"--SOLVER.OPTIMIZER_NAME",type=str,
                        help="optimizer (default:SGD)")
    parser.add_argument("-criterion","--SOLVER.CRITERION",type=str,
                        help="Loss Function (default: GIoU_L1Loss)")
    parser.add_argument("-lr","--SOLVER.LEARNING_RATE",type=float,
                        help="Learning rate (default:0.01)")
    parser.add_argument('-patience','--SOLVER.LR_SCHEDULER_PATIENCE',type=int,
                        help='Number of events to wait if no improvement and then stop the training. (default:100)')
    parser.add_argument('-factor','--SOLVER.LR_SCHEDULER_FACTOR',type=float,
                        help='factor of lr_scheduler (default: 1/3)')
    parser.add_argument("-lr_scheduler","--SOLVER.LR_SCHEDULER",type=str)


    # OUTPUT 
    parser.add_argument("-n_saved","--OUTPUT.N_SAVED",type=int)

    # TOOLS
    parser.add_argument("-image_n","--TOOLS.image_n",type=int,default=3)
    parser.add_argument("-save_path","--TOOLS.save_path",type=str)

    arg = parser.parse_args()
    return arg

def merge_from_dict(cfg,arg_dict):
    for key in arg_dict:
        if arg_dict[key] != None:
            cfg.merge_from_list([key,arg_dict[key]])
    return cfg

def seed_torch(seed=15):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if  __name__ == "__main__":
    seed_torch()
    arg = load_arg()

    if arg.CONFIG_FILE != None:
        cfg.merge_from_file(arg.CONFIG_FILE)

    cfg = merge_from_dict(cfg,vars(arg))
    print(cfg)

    if cfg.OUTPUT.DIR_NAME and not path.exists(cfg.OUTPUT.DIR_NAME):
        mkdir(cfg.OUTPUT.DIR_NAME)
    
    train(cfg)