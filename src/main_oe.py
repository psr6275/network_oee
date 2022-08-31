import argparse
import logging
import os, pickle, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Train models")
parser.add_argument(
    "--dataset",
    default="2017",
    type=str,
    help="dataset name: 2017",
)

parser.add_argument(
    "--exp",
    default=1,
    type=int,
    help="experiment number should be between 1~3",
)

# parser.add_argument(
#     "--VAN",
#     default=False,
#     action="store_true",
#     help="Vanilla training without OE",
# )

# parser.add_argument(
#     "--LS",
#     default=False,
#     action="store_true",
#     help="label smoothing during training",
# )
parser.add_argument(
    "--data-dir",
    default="../data/ids-dataset",
    type=str,
    metavar="PATH",
    help="data directory for the cicids datasets",
)
parser.add_argument(
    "--result-dir",
    default="../results/cicidsv3.0",
    type=str,
    metavar="PATH",
    help="result directory for the trained model and data",
)
parser.add_argument(
    "--batch-size",
    default=256,
    type=int,
    help="batch size for training",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    help="training epochs",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="gpu device information",
)

def _run_training(args):
    from utils import load_cicids_binary_data, load_cicids_mult_data
    from utils import load_model_path

    device = torch.device(args.device)
    result_dir = args.result_dir
    batch_size = args.batch_size
    epochs = args.epochs
    LS = True
    OE = True

    assert args.dataset in ['2017', '2018']

    if args.dataset =="2017":
        from main_utils import exp_label_2017
        lab_cluster, lab_dic, lab_name, ooc_cols = exp_label_2017(args.exp)   
        from main_utils import lab_2017 as lab_name_tot
    else:
        from main_utils import exp_label_2018
        lab_cluster, lab_dic, lab_name, ooc_cols = exp_label_2018(args.exp)
        from main_utils import lab_2018 as lab_name_tot    
    
    cicids_bn = load_cicids_binary_data(args.dataset,lab_cluster, lab_name, result_dir,True, None)
    cicids_m = load_cicids_mult_data(args.dataset,lab_dic, lab_name,result_dir,True, None)
        
    bn_save_model, mul_save_model = load_model_path(args.dataset, lab_name, epochs, None, LS, OE)

    if os.path.isfile(os.path.join(result_dir, bn_save_model)):
        logging.info("Already exists: "+ bn_save_model)
        # print("Already exists: ", bn_save_model)
    else:
        from main_utils import binary_training
        logging.info("Train model to: "+ bn_save_model)
        # print("Train model to: ", bn_save_model)
        binary_training(cicids_bn, epochs, batch_size, device, result_dir, bn_save_model)
    
    if os.path.isfile(os.path.join(result_dir, mul_save_model)):
        logging.info("Already exists: "+ mul_save_model)
        # print("Already exists: ", mul_save_model)
    else:
        from main_utils import mult_training
        logging.info("Train model to: "+ mul_save_model)
        # print("Train model to: ", mul_save_model)
        mult_training(cicids_bn, cicids_m, epochs, batch_size, device, result_dir, mul_save_model, LS, OE)


if __name__ =="__main__":

    args = parser.parse_args()
    log_file = os.path.join(args.result_dir, "train_cicids%s_epochs_%d_EXP_%s.log"%(args.dataset, args.epochs, args.exp))

    
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    _run_training(args)
    
    
    