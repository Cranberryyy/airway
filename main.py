import  tensorflow as tf
import numpy as np
from config import get_config
from datasets import get_dataset 
import argparse
from utils import gather_info_training, gather_info_testing, update_dataset_opt, gather_info_val
from models import get_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
            print(e)

"""
Training phase parameters setting
"""
parser = argparse.ArgumentParser(description="Train an segmentation network")
parser.add_argument("--run_name", default='exact', help="Which dataset will be used in the training")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--pool_size", type=int, default=64, help="Batch size")
parser.add_argument("--config", default="config/basemodel.yaml", help="run configs")

"""model setting"""
parser.add_argument("--net", type=str, default='vnet', help="the network to train")
parser.add_argument("--model", type=str, default='basemodel', help="the model seeting")
parser.add_argument("--model_stage", type=int, default=1, help="1: one header; 2: two headers")
parser.add_argument("--train_phase", type=int, default=1, help="1: voxel; 2: point cloud; 3: merge; 4: jointly")


"""training setting"""
parser.add_argument("--epoch", type=int, default=500, help="Batch size")
parser.add_argument("--cropping_size", type=list, default=[96, 96, 96],
                     help="cropping_size")
parser.add_argument("--slide_window_step_train", type=list, default=[64, 64, 64], help= "the size of slide window in training")
parser.add_argument("--slide_window_step", type=list, default=[64, 64, 64], 
                    help="the interleave of slide window, if any > 4, then means the size of slide window")
parser.add_argument("--loader_case", type=int, default=8, help="the number of cases loading in the loader")
parser.add_argument("--sample_case", type=int, default=40, help="the number of cases sampled in one training volumes in each epoch")
parser.add_argument("--load_coord", type=bool, default=True, help="whether to load the coordinate")
parser.add_argument("--continue_train", type=bool, default=False, 
                    help='whether to continue train, if True, please specify the path the ckpt to restore')
parser.add_argument("--resume_model", type=bool, default=False, 
                    help='whether to continue train, if True, please specify the path the ckpt to restore')
parser.add_argument("--reset_head2", type=bool, default=False, 
                    help='whether to continue train, if True, please specify the path the ckpt to restore')
parser.add_argument("--model_dir", type=str, default="f_model_saved", help="Where to save the training log and model")
parser.add_argument("--resume_dir", type=str, default=None, help="the path of ckpt to restore")
parser.add_argument("--val_epoch", type=int, default=100, help="the step to val the model")

parser.add_argument("--train_server", default=False, action='store_true', 
                        help='If True, please specify the number of workers in dataloader')
parser.add_argument("--num_workers", type=int, default=4, help="the number of workers in dataloader")
parser.add_argument("--mode", type=str, default='Train_Val', choices=['Train', 'Train_Val', 'Val', 'Test'],
                        help='The status of the model')

parser.add_argument("--data_stage", type=int, default=1, choices=[1,2, 3],
                    help="1: train simple case; 2: train densely; 3: training on the tree")


parser.add_argument("--random_seed", type=int, default=1, help="the random seed to split val from train")
parser.add_argument("--pretrain", default=True, action='store_true', help='whether to continue train')
parser.add_argument("--prepare_data", type=bool, default=False, 
                        help='whether to prepare data, i.e., case name in a file')
parser.add_argument("--save_prepare_data", type=bool, default=False, 
                        help='whether to save case name file in the folder')
parser.add_argument("--save_val", type=bool, default=True, 
                        help='whether to validating results')

"""learning related setting"""
parser.add_argument("--learning_rate_head1", type=float, default=0.0001, help="learning rate for (general case) head1")
parser.add_argument("--learning_rate_head2", type=float, default=0.0001, help="learning rate of (hard case) head2")
parser.add_argument("--optimizer", type=str, default='adam', help="the optimizer to train")
parser.add_argument("--train_voxel", type=bool, default=False, 
                    help='whether to continue train, if True, please specify the path the ckpt to restore')
parser.add_argument("--train_point", type=bool, default=False, 
                    help='whether to continue train, if True, please specify the path the ckpt to restore')
parser.add_argument("--out_loss_1", type=str, default='wce', help="the loss function to the final output")
parser.add_argument("--out_loss_2", type=str, default='wce', help="the loss function to the final output")
parser.add_argument("--deep_loss_1", type=str, default='tversky', help="the loss function to the deep supervision")
parser.add_argument("--deep_loss_2", type=str, default='tversky', help="the loss function to the deep supervision")
parser.add_argument("--w_final_loss", type=float, default=10, help="the weight of final loss, voxel-wise")
parser.add_argument("--w_connec_loss", type=float, default=1, help="the weight of connectivity loss")
parser.add_argument("--w_mid_loss", type=float, default=0.5, help="the weight of middle loss")
parser.add_argument("--w_distance_loss", type=float, default=1, help="the weight of distance loss")
# parser.add_argument("--warmup_learning_rate", type=float, default=1, help="Start to inverse-decay learning rate at this step.")
# parser.add_argument("--warmup_portion", type=float, default=0.001, help="How many steps we inverse-decay learning.")
# parser.add_argument("--warmup_steps", type=int, default=0, help="Depends on portion, no need to specify.")

"""parameters for using point clouds"""
parser.add_argument("--num_sampled_point_pred", type=int, default=2048, help='the number of sampled points in prediction')
parser.add_argument("--num_sampled_point_gt", type=int, default=2048, help='the number of sampled points in ground truth')
parser.add_argument("--point_net", type=str, default='cascased_point_completion', help="the network for point clouds")
args = parser.parse_args()



def main():
    np.random.seed(args.random_seed)
    opts = get_config(args.config)
    dataset_opt = opts['dataset'][args.run_name]
    dataset_opt = update_dataset_opt(dataset_opt, args)
    net_opt = opts['net_setting']
    
    # if args.prepare_data:
    #     if args.run_name == 'atm':
    #         prepare_atm(dataset_opt, args.save_prepare_data)
    #     elif args.run_name == 'exact':
    #         prepare_exact(dataset_opt, args.save_prepare_data)
    #     else:
    #         raise ValueError("Wrong dataset name: %s" %(args.run_name))

    Model = get_model(args.model)(args, net_opt)
    if args.continue_train or args.resume_model:
        Model.resume(args)

    if args.mode == 'Train': 
        train_dataset = get_dataset(dataset_opt, args, 'Train')
        train_loader = train_dataset.data_loader(args)
        info = gather_info_training(args, train_dataset=train_dataset)
    elif args.mode == 'Train_Val':
        train_dataset = get_dataset(dataset_opt, args, 'Train')
        val_dataset = get_dataset(dataset_opt, args, 'Val')
        val_loader = val_dataset.data_loader(args)
        train_loader = train_dataset.data_loader(args)
        info = gather_info_training(args, train_dataset=train_dataset, val_dataset=val_dataset)
    elif args.mode == 'Val':
        val_dataset = get_dataset(dataset_opt, args, 'Val')
        val_loader = val_dataset.data_loader(args)
        info = gather_info_val(args, val_dataset=val_dataset)
    elif args.mode == 'Test':
        test_dataset = get_dataset(dataset_opt, args, 'Test')
        test_loader = test_dataset.data_loader(args)
        info = gather_info_testing(args, test_dataset=test_dataset)
    else:
        raise ValueError("Wrong mode: %s" %(args.mode))


    if args.mode == 'Train':
        Model.train(args, train_loader, info)
    elif args.mode == 'Train_Val':
        Model.train_val(args, train_loader, val_loader, info)
    elif args.mode == 'Val':
        Model.val(args, val_loader, info)
    elif args.mode == 'Test':
        Model.test(args, test_loader, info)
    else:
        raise ValueError("Wrong mode: %s" %(args.mode))
    


if __name__ == "__main__":
    main()

