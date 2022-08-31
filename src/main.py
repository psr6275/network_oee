import argparse
import logging
import os, pickle, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import make_dataloader, train_model, test_model, train_model_with_oe_KL

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
    "--OE",
    default=False,
    action="store_true",
    help="outlier exposure during attack type classification",
)

parser.add_argument(
    "--OEE",
    default=False,
    action="store_true",
    help="outlier exposure ensemble for mult-class classification",
)
parser.add_argument(
    "--VAN",
    default=False,
    action="store_true",
    help="Vanilla training in OEE",
)

parser.add_argument(
    "--LS",
    default=False,
    action="store_true",
    help="label smoothing during training",
)
parser.add_argument(
    "--result-dir",
    default="../results/cicidsv1.3",
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
def binary_training(cicids_bn, epochs, batch_size, device, result_dir, save_model):
    from utils import BinaryNN
    Xtr_bn = cicids_bn[-1].transform(cicids_bn[0])
    Xte_bn = cicids_bn[-1].transform(cicids_bn[2])
    ytr_bn = cicids_bn[1]
    yte_bn = cicids_bn[3]

    train_bnloader = make_dataloader(Xtr_bn,ytr_bn, batch_size = batch_size, shuffle=True)
    test_bnloader = make_dataloader(Xte_bn,yte_bn, batch_size = batch_size, shuffle=False)

    n_features = Xtr_bn.shape[1]
    clf_bn = BinaryNN(n_features)
    criterion_bn = nn.BCELoss()
    optim_bn = optim.Adam(clf_bn.parameters(),lr=0.0001)    

    clf_bn = train_model(clf_bn, train_bnloader, optim_bn, device, criterion_bn, epochs, save_dir = result_dir, 
                     save_model = save_model, binary=True)
    test_res = test_model(clf_bn, test_bnloader, criterion_bn, device, 100.0, binary=True)
    logging.info("test performance for "+save_model)
    logging.info("\nloss {}\t"
                "Prec@1 test {:.3f} ({:.3f})\t".format(
                    test_res[0],
                    test_res[1],
                    test_res[2]
                )
    )
    # print("test performance for ", save_model)
    # print(test_model(clf_bn, test_bnloader, criterion_bn, device, 100.0, binary=True))

def mult_oee_training(cicids_m, epochs, batch_size, device, result_dir, save_model, LS, van=False):
    
    from utils import MultNN
    
    Xtrm = cicids_m[-1].transform(cicids_m[0])
    Xtem = cicids_m[-1].transform(cicids_m[2])
    ytrm = cicids_m[1]
    ytem = cicids_m[3]
    Xtroe = cicids_m[-1].transform(cicids_m[4])
    Xooc = cicids_m[-1].transform(cicids_m[6])
    
    n_features = Xtrm.shape[1]
    num_class = len(np.unique(ytrm))  
    n_hidden = 32
    
    train_mulloader = make_dataloader(Xtrm,ytrm.to_numpy().flatten(), batch_size = batch_size, shuffle=True)
    test_mulloader = make_dataloader(Xtem,ytem.to_numpy().flatten(), batch_size = batch_size, shuffle=False)
    outlier_trloader = make_dataloader(Xtroe,np.zeros(len(Xtroe)),batch_size=batch_size, shuffle=True)

    clf_mul = MultNN(n_features, n_hidden=n_hidden, num_class=num_class)  
    optim_mul = optim.Adam(clf_mul.parameters(),lr=0.0001)

    if LS:
        criterion_mul = nn.CrossEntropyLoss(label_smoothing=0.01)
    else:
        criterion_mul = nn.CrossEntropyLoss()

    if van:    
       clf_mul = train_model(clf_mul, train_mulloader, optim_mul, device, criterion_mul, epochs, save_dir = result_dir, 
                     save_model = save_model, binary=False)
    else:
        criterion_oe = nn.KLDivLoss()

        clf_mul = train_model_with_oe_KL(clf_mul, train_mulloader, outlier_trloader, num_class, optim_mul, device, 
                                    criterion_mul, criterion_oe, 1.0, epochs, save_dir = result_dir, 
                        save_model = save_model,binary=False)
    
    test_res = test_model(clf_mul, test_mulloader, criterion_mul, device, 100.0, binary=False)
    logging.info("test performance for "+ save_model)
    logging.info("\nloss {}\t"
                "Prec@1 test {:.3f} ({:.3f})\t".format(
                    test_res[0],
                    test_res[1],
                    test_res[2]
                )
    )

def mult_training(cicids_bn, cicids_m, epochs, batch_size, device, result_dir, save_model, LS, OE):
    
    from utils import MultNN
    
    Xtrm = cicids_m[-1].transform(cicids_m[0])
    Xtem = cicids_m[-1].transform(cicids_m[2])
    ytrm = cicids_m[1]
    ytem = cicids_m[3]

    train_mulloader = make_dataloader(Xtrm,ytrm.to_numpy().flatten(), batch_size = batch_size, shuffle=True)
    test_mulloader = make_dataloader(Xtem,ytem.to_numpy().flatten(), batch_size = batch_size, shuffle=False)

    num_class = len(np.unique(ytrm))
    n_features = Xtrm.shape[1]

    clf_mul = MultNN(n_features, n_hidden=32, num_class=num_class)
    optim_mul = optim.Adam(clf_mul.parameters(),lr=0.0001)

    if LS:
        criterion_mul = nn.CrossEntropyLoss(label_smoothing=0.01)
    else:
        criterion_mul = nn.CrossEntropyLoss()

    if OE:
        Xbtr = cicids_m[-1].transform(cicids_bn[0][cicids_bn[1]==0])        
        outlier_trloader = make_dataloader(Xbtr,np.zeros(len(Xbtr)),batch_size=batch_size, shuffle=True)
        criterion_oe = nn.KLDivLoss()

        clf_mul = train_model_with_oe_KL(clf_mul, train_mulloader, outlier_trloader, num_class, optim_mul, device, 
                                 criterion_mul, criterion_oe, 1.0, epochs, save_dir = result_dir, 
                     save_model = save_model,binary=False)
    else:
        clf_mul = train_model(clf_mul, train_mulloader, optim_mul, device, criterion_mul, epochs, save_dir = result_dir, 
                     save_model = save_model, binary=False)
    
    test_res = test_model(clf_mul, test_mulloader, criterion_mul, device, 100.0, binary=False)
    logging.info("test performance for "+ save_model)
    logging.info("\nloss {}\t"
                "Prec@1 test {:.3f} ({:.3f})\t".format(
                    test_res[0],
                    test_res[1],
                    test_res[2]
                )
    )
    # print("test performance for ", save_model)
    # print(test_model(clf_mul, test_mulloader, criterion_mul, device, 100.0, binary=False))


def _run_training(args):
    from utils import cluster_labels_2017, cluster_labels_2018, load_cicids_binary_data, load_cicids_mult_data
    from utils import load_model_path

    device = torch.device(args.device)
    result_dir = args.result_dir
    batch_size = args.batch_size
    epochs = args.epochs
    LS = args.LS
    OE = args.OE

    assert args.dataset in ['2017', '2018']

    if args.dataset =="2017":
        lab_dic, lab_name = cluster_labels_2017()       
    else:
        lab_dic, lab_name = cluster_labels_2018()
    
    ooc_list = [None]+list(np.arange(len(lab_name)))

    for ooc_cols in ooc_list:
        cicids_bn = load_cicids_binary_data(args.dataset,result_dir,True, ooc_cols)
        cicids_m = load_cicids_mult_data(args.dataset,result_dir,True, ooc_cols)
        
        bn_save_model, mul_save_model = load_model_path(args.dataset, lab_name, epochs, ooc_cols, LS, OE)

        if os.path.isfile(os.path.join(result_dir, bn_save_model)):
            logging.info("Already exists: "+ bn_save_model)
            # print("Already exists: ", bn_save_model)
        else:
            logging.info("Train model to: "+ bn_save_model)
            # print("Train model to: ", bn_save_model)
            binary_training(cicids_bn, epochs, batch_size, device, result_dir, bn_save_model)
        
        if os.path.isfile(os.path.join(result_dir, mul_save_model)):
            logging.info("Already exists: "+ mul_save_model)
            # print("Already exists: ", mul_save_model)
        else:
            logging.info("Train model to: "+ mul_save_model)
            # print("Train model to: ", mul_save_model)
            mult_training(cicids_bn, cicids_m, epochs, batch_size, device, result_dir, mul_save_model, LS, OE)

def _run_oee_training(args):
    from utils import cluster_labels_2017, cluster_labels_2018, load_cicids_binary_data, load_cicids_mult_data
    from utils import load_oee_model_path, load_cicids_OEE_mult_data
    import copy

    device = torch.device(args.device)
    result_dir = args.result_dir
    batch_size = args.batch_size
    epochs = args.epochs
    LS = args.LS
    van = args.VAN

    assert args.dataset in ['2017', '2018']

    if args.dataset =="2017":
        lab_dic, lab_name = cluster_labels_2017(include_benign=True)       
    else:
        lab_dic, lab_name = cluster_labels_2018(include_benign=True)
    
    ooc_list = list(np.arange(len(lab_name)))[1:]
    for ooc_cols in ooc_list:
        oee_list = list(np.arange(len(lab_name)))
        oee_list.pop(ooc_cols)
        for oee_cols in oee_list:
            cicids_m = load_cicids_OEE_mult_data(args.dataset, result_dir, ooc_cols=ooc_cols, oee_cols=oee_cols)
            mul_save_model = load_oee_model_path(args.dataset, lab_name, epochs, ooc_cols, oee_cols, LS, van)
            if os.path.isfile(os.path.join(result_dir, mul_save_model)):
                logging.info("Already exists: "+ mul_save_model)
                # print("Already exists: ", mul_save_model)
            else:
                logging.info("Train model to: "+ mul_save_model)
                # print("Train model to: ", mul_save_model)
                mult_oee_training(cicids_m, epochs, batch_size, device, result_dir, mul_save_model, LS, van)
    

if __name__ =="__main__":

    args = parser.parse_args()
    log_file = os.path.join(args.result_dir, "train_cicids%s_epochs_%d_OEE_%s_OE_%s_LS_%s_VAN_%s.log"%(args.dataset,args.epochs,args.OEE,args.OE, args.LS, args.VAN))
    
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    
    # orig_logging_level = logging.getLogger().level
    # logging.getLogger().setLevel(logging.INFO)
    # file_handler = logging.FileHandler(os.path.join(args.result_dir, log_file))
    # logging.getLogger().addHandler(file_handler)
    
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    if args.OEE:
        _run_oee_training(args)
    else:
        _run_training(args)
    
    # logging.getLogger().setLevel(orig_logging_level)

    
    # formatter = logging.Formatter()
    # file_handler = logging.FileHandler(os.path.join(args.result_dir, log_file))
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    