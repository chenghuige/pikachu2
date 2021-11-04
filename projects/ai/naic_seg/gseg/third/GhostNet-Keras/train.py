# -- coding: utf-8 --

from data_load.data_load import Dataload
from data_load.mnist_load import MnistLoad
from model.GhostNet import GhostModel
from trainer.ghost_trainer import GhostTrainer
from config.config import get_config_from_json, get_train_args


def train_evaluate():

    print('*************解析配置*************\n')
    
    parser = None
    config = None

    try:
        args, parser = get_train_args()
        config,_ = get_config_from_json(args.config)
    except Exception as e:
        print('[ERROR] 配置无效, %s' % e)
        if parser:
            parser.print_help()
        print('[Attention] 参考: python train.py -c config/ghost_config.json')
        exit(0)

    #训练参数解析
    epoch = config.num_epochs
    batchsize = config.batch_size
    numclass = config.num_class
    size = config.size
    use_mnist_data = config.use_mnist_data
    train_data_list = config.train_list
    test_data_list = config.test_list

    print('*************加载数据*************')
    if(use_mnist_data):
        dl = MnistLoad(numclass, size)
    else:
        dl = Dataload(train_data_list, test_data_list, numclass, size)
    
    train_data = dl.get_train_data()
    test_data = dl.get_test_data()
    print('*************加载完成*************\n')



    print('*************构造网络*************')
    if(use_mnist_data):
        model = GhostModel(numclass,size,1)
    else:
        model = GhostModel(numclass,size,3)
    print('*************构造完成*************\n')



    print('*************训练网络*************')
    trainer = GhostTrainer(
        model.model,
        [train_data, test_data],
        batchsize,
        epoch)
    trainer.train()
    print('*************训练完成*************\n')



    print('*************评估网络*************')
    trainer.test()
    print('*************评估完成*************\n')


if __name__ == '__main__':
    train_evaluate()
