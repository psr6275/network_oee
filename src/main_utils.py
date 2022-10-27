import os, pickle, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import logging

from utils import make_dataloader, train_model_with_oe_KL, train_model, test_model

lab_2017 = ['Benign','Web Attack \x96 Brute Force', 'Web Attack \x96 XSS', 'Web Attack \x96 Sql Injection',\
            'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest',\
            'DDoS', 'Bot', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'Heartbleed', 'Infiltration']
labdic_2017 = {'Benign':0,'Web Attack \x96 Brute Force':1, 'Web Attack \x96 XSS':2, 'Web Attack \x96 Sql Injection':3,\
            'DoS Hulk':4, 'DoS GoldenEye':5, 'DoS slowloris':6, 'DoS Slowhttptest':7,\
            'DDoS':8, 'Bot':9, 'PortScan':10, 'FTP-Patator':11, 'SSH-Patator':12, 'Heartbleed':13, 'Infiltration':14}
lab_2018 = ['Benign','Brute Force -Web', 'Brute Force -XSS', 'SQL Injection', 'DoS attacks-Hulk', \
            'DoS attacks-GoldenEye','DoS attacks-Slowloris', 'DoS attacks-SlowHTTPTest', \
            'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'Bot','Infilteration',\
            'FTP-BruteForce','SSH-Bruteforce']   
labdic_2018 = {'Benign':0,'Brute Force -Web':1, 'Brute Force -XSS':2, 'SQL Injection':3, 'DoS attacks-Hulk':4, \
            'DoS attacks-GoldenEye':5,'DoS attacks-Slowloris':6, 'DoS attacks-SlowHTTPTest':7, \
            'DDoS attacks-LOIC-HTTP':8, 'DDOS attack-HOIC':9, 'Bot':10,'Infilteration':11,\
            'FTP-BruteForce':12,'SSH-Bruteforce':13}   
            
def exp_label_2017(exp_num=1):
    assert exp_num in [1,2,3,4]
    lab_cluster = {0: [1,2], 1: [4,5,6,7], 2: [8], 3: [9], 4: [10], 5: [11,12]}
    lab_name = ['Web_Attack', 'DoS_attacks', 'DDoS_attacks', 'Bot', 'PortScan', 'Bruteforce']
    
    if exp_num==1:
        ooc_class = 3        
    elif exp_num==2:
        ooc_class = 13
    elif exp_num ==3:
        ooc_class = 14
    else:
        ooc_class = 11
        lab_cluster[5].remove(ood_class)
    
    lab_dic = {}
    for nlab in lab_cluster:
        for lab in lab_cluster[nlab]:
            lab_dic[lab_2017[lab]] = nlab
    print(lab_dic)
    return lab_cluster, lab_dic, lab_name, ooc_class

def exp_label_2018(exp_num=1):
    assert exp_num in [1,2,3]
    lab_cluster = {0: [1,2], 1: [4,5,6], 2: [8,9], 3: [10], 4: [11], 5: [13]}
    lab_name = ['Web_Attack', 'DoS_attacks', 'DDoS_attacks', 'Bot', 'Infilteration', 'Bruteforce']
    
    if exp_num==1:
        ooc_class = 3       
    elif exp_num==2:
        ooc_class = 7
    else:
        ooc_class = 12

    lab_dic = {}
    for nlab in lab_cluster:
        for lab in lab_cluster[nlab]:
            lab_dic[lab_2018[lab]] = nlab
    print(lab_dic)
    return lab_cluster, lab_dic, lab_name, ooc_class
    
def split_df_with_label(df, lab_col='Label'):
    df_label = df[lab_col]
    lab_names = df_label.unique()

    split_df = {}

    for lab in lab_names:
        idx = df_label==lab
        split_df[lab] = df[idx].loc[:,df.columns!=lab_col]
    return split_df

def make_splitted_dataloader(df, lab_col, bn_scaler, mul_scaler,lab_dic, device, batch_size):
    from utils import make_dataloader
    df_label = df[lab_col]
    lab_names = df_label.unique()
    split_df = split_df_with_label(df, lab_col)

    bndl = {}
    muldl = {}
    # if dataset == '2017':
    #     labdic_orig = labdic_2017
    # else:
    #     labdic_orig =  labdic_2018

    for lab in lab_names:
        if lab =='Benign':
            bny = np.zeros(split_df[lab].shape[0])
            muly = -np.ones(split_df[lab].shape[0])
        elif lab in lab_dic.keys():
            # seen attack
            # print(lab_dic[lab])
            bny = np.ones(split_df[lab].shape[0])
            muly = lab_dic[lab]*np.ones(split_df[lab].shape[0])
        else:
            # unseen attack            
            bny = np.ones(split_df[lab].shape[0])
            muly = -np.ones(split_df[lab].shape[0])

        bndl[lab] = make_dataloader(split_df[lab], bny, batch_size, False, bn_scaler)
        muldl[lab] = make_dataloader(split_df[lab], muly, batch_size, False, mul_scaler)        

    return bndl, muldl


    

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
