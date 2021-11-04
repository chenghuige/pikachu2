import argparse



def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Sogou Wide and Deep\'s arguments parser')

    # Load & Save
    #parser.add_argument('--train_data_dir', type=str, default='/dataset/wide_n_deep/ofrecord/train/',
    #parser.add_argument('--train_data_dir', type=str, default='/dataset/wide_n_deep/ofrecord/repeat_1w',
    #parser.add_argument('--train_data_dir', type=str, default='/dataset/wide_n_deep/ofrecord/train_1w',
    parser.add_argument('--train_data_dir', type=str, default='/dataset/wide_n_deep/ofrecord/train/',
                        help='data path')
    parser.add_argument('--pretrain_model_path', type=str, default='/dataset/wide_n_deep/models/of_model',
                        help='Pretrain model save path')
    parser.add_argument('--model_save_path', type=str, default='./save/',
                        help='Model save path')
    parser.add_argument('--loss_print_steps', type=int, default=100,
                        help='print loss after n batches train')
    parser.add_argument('--save_snapshot_after_batch_size', type=int, default=2000,
                        help='Save snapshot after n batches train')
    parser.add_argument('--data_part_num', type=int, default=256,
                        help='Number of train dataset oneflow record files')
    # Train
    parser.add_argument('--max_steps', type=int, default=92000,
                        help='Max of training steps.')
    parser.add_argument('--max_feat_len', type=int, default=400,
                        help='Max of feature length')
    parser.add_argument('--batch_size_per_device', type=int, default=500,
                        help='batch size per device/gpu')
    parser.add_argument('--device_num', type=int, default=1,
                        help='Device number per node')
    parser.add_argument('--node_num', type=int, default=1,
                        help='node/machine number for training') #TODO config ip
    parser.add_argument('--feature_dict_size', type=int, default=2274370,
                        help='Feature Dict Size')
    parser.add_argument('--field_size', type=int, default=94,
                        help='Field Size')
    parser.add_argument('--embedding_size', type=int, default=500,
                        help='Deep Embedding Size')
    parser.add_argument('--deep_layers', type=list, default=[47000, 50],
                        help='Deep FC layers')
    #parser.add_argument('--embedding_size', type=int, default=8,
    #                    help='Deep Embedding Size')
    #parser.add_argument('--deep_layers', type=list, default=[752, 512],
    #                    help='Deep FC layers')
    parser.add_argument('--enable_model_split', type=bool, default=True,
                        help='enable model split')
    # Predict
    parser.add_argument('--max_predict_steps', type=int, default=331,
                        help='Max of prediction steps.')
    parser.add_argument('--predict_data_dir', type=str, default='/dataset/wide_n_deep/ofrecord/predict/',
                        help='data path')
    parser.add_argument('--predict_model_path', type=str,
                        default='/dataset/wide_n_deep/models/snapshot_90001', help='')
    parser.add_argument('--predict_data_part_num', type=int, default=256,
                        help='Number of predict dataset oneflow record files')

    # Learning rate strategy
    parser.add_argument('--primary_lr', type=float, default=1e-3,
                        help='Primary learning rate')
    parser.add_argument('--lr_decay_batches', type=int, default=10000,
                        help='Update learning rate after n batches train')
    parser.add_argument('--min_lr', type=float, default=0.0,
                        help='Minimum value of learning rate')
    # warm up
    parser.add_argument('--warmup_batches', type=int, default=1000,
                        help='Number of batches the warmup needs')
    parser.add_argument('--start_multiplier', type=float, default=0,
                        help='Warmup start multiplier')
    parser.add_argument('--weight_l2', type=float, default=1e-2,
                        help='Weight L2')

    return parser


if __name__ == '__main__':
    parser = get_parser(None)
    config = parser.parse_known_args()
    print(config)
