import argparse
import logging
import os

from lichee import plugin

from module.models import ConcatCls, EmbeddingTrainer
from module.feature_parser import FrameFeature, TagParser, VidParser
from module.utils import LayeredOptim, BCELoss, PRScore


def parse_args():
    parser = argparse.ArgumentParser(description="Tencent Kandian video embedding challenge")

    parser.add_argument('--mode', type=str, default='train', help='choose mode from [train, test, eval]')
    parser.add_argument("--trainer", type=str,
                        help="choose trainer, you can run --show_trainer list all supported trainer")
    parser.add_argument("--model_config_file", type=str, help="The path of configuration json file.")
    parser.add_argument("--show", type=str,
                        help="list all supported module, [trainer|target|model|loss|optim|lr_schedule]")
    parser.add_argument("--dataset", type=str, default='EVAL_DATA')
    parser.add_argument('--checkpoint', type=str, default='Epoch_3.bin')
    parser.add_argument('--to-save-file', type=str, default='res.json')

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    args = parse_args()
    trainer = plugin.get_plugin(plugin.PluginType.TRAINER, args.trainer)
    if args.mode == 'train':
        trainer = trainer(args.model_config_file)
        trainer.train()
    elif args.mode == 'test':
        trainer = trainer(args.model_config_file, init_model=False)
        if os.path.exists(args.to_save_file):
            raise FileExistsError('to save file: {} already existed'.format(args.to_save_file))
        trainer.evalute_checkpoint(
            dataset_key=args.dataset, checkpoint_file=args.checkpoint, to_save_file=args.to_save_file)
    else:
        trainer = trainer(args.model_config_file, init_model=False)
        trainer.evaluate_spearman(dataset_key=args.dataset, checkpoint_file=args.checkpoint)
