import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def set_binary_label(df, label_col, default_class='BENIGN', return_col = False, 
                     ooc_cols = None, lab_cluster = None):
    df_y = df[label_col].copy()
    rep_class = {}
    lab_names = df_y.unique()
    if ooc_cols is not None:
        assert lab_cluster is not None
        print("ooc data: ", lab_cluster[ooc_cols])
        ooc_labs = lab_cluster[ooc_cols]
    else:
        ooc_labs = []
        
    drop_idx = pd.Index([])    
    
    for lab in lab_names:
        if lab == default_class:
            rep_class[lab] = 0
        else:
            if lab in ooc_labs:
#                 print(df[df[label_col]==lab].index)
                drop_idx = drop_idx.append(df[df[label_col]==lab].index)
                rep_class[lab] = -1
            else:
                rep_class[lab] = 1
                
    if return_col:
        return df_y.replace(rep_class).drop(drop_idx)
    else:
        return df.replace({label_col:rep_class}).drop(drop_idx)
    
def cluster_labels_2018(return_cluster = False, include_benign = False):
    if include_benign:
        lab_cluster = {0: ['Benign'],
                1: ['Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'],
                2: ['DoS attacks-Hulk', 'DoS attacks-GoldenEye' ],
                3: ['DoS attacks-Slowloris', 'DoS attacks-SlowHTTPTest'],
                4: ['DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC'],
                5: ['Bot'],
                6: ['Infilteration'],
                7: ['FTP-BruteForce'],
                8: ['SSH-Bruteforce']   
                }
        lab_name = ['Benign','Web_Attack','DoS_attacks-type1','DoS_attacks-type2','DDoS_attacks','Bot','Infilteration',
                 'FTP-BruteForce','SSH-Bruteforce']
    else:
        lab_cluster = {0: ['Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'],
                    1: ['DoS attacks-Hulk', 'DoS attacks-GoldenEye' ],
                    2: ['DoS attacks-Slowloris', 'DoS attacks-SlowHTTPTest'],
                    3: ['DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC'],
                    4: ['Bot'],
                    5: ['Infilteration'],
                    6: ['FTP-BruteForce'],
                    7: ['SSH-Bruteforce']   
        }
        lab_name = ['Web_Attack','DoS_attacks-type1','DoS_attacks-type2','DDoS_attacks','Bot','Infilteration',
                    'FTP-BruteForce','SSH-Bruteforce']
    if return_cluster:
        return lab_cluster, lab_name
    else:
        lab_dic = {}
        for nlab in lab_cluster:
            for lab in lab_cluster[nlab]:
                lab_dic[lab] = nlab
        print(lab_dic)
        return lab_dic, lab_name

def cluster_labels_2017(return_cluster = False, include_benign = False):
    if include_benign:
        lab_cluster = {0: ['Benign'],
                    1: ['Web Attack \x96 Brute Force', 'Web Attack \x96 XSS', 'Web Attack \x96 Sql Injection'],
                    2: ['DoS Hulk', 'DoS GoldenEye' ],
                    3: ['DoS slowloris', 'DoS Slowhttptest'],
                    4: ['DDoS'],
                    5: ['Bot'],                       
                    6: ['PortScan'],
                    7: ['FTP-Patator'],
                    8: ['SSH-Patator']    
        }
        lab_name = ['Benign','Web_Attack','DoS_attacks-type1','DoS_attacks-type2','DDoS_attacks','Bot','PortScan',
                    'FTP-BruteForce','SSH-Bruteforce']
    else:
        lab_cluster = {0: ['Web Attack \x96 Brute Force', 'Web Attack \x96 XSS', 'Web Attack \x96 Sql Injection'],
                1: ['DoS Hulk', 'DoS GoldenEye' ],
                2: ['DoS slowloris', 'DoS Slowhttptest'],
                3: ['DDoS'],
                4: ['Bot'],                       
                5: ['PortScan'],
                6: ['FTP-Patator'],
                7: ['SSH-Patator']    
                }
        lab_name = ['Web_Attack','DoS_attacks-type1','DoS_attacks-type2','DDoS_attacks','Bot','PortScan',
                 'FTP-BruteForce','SSH-Bruteforce']
    if return_cluster:
        return lab_cluster, lab_name
    else:
        lab_dic = {}
        for nlab in lab_cluster:
            for lab in lab_cluster[nlab]:
                lab_dic[lab] = nlab
        print(lab_dic)
        return lab_dic, lab_name
       
        
def split_ooc_data(df, lab_name, ooc_cols):
    # ooc_cols should be index for lab name! it is because labels in df are already transformed into integer index!
    print("ooc column: ", lab_name[ooc_cols])
    ooc_idx = df[df['Label']==ooc_cols].index        
    lab_map = {}
    j=0
    for i,lab in enumerate(lab_name):
        if i == ooc_cols:
            lab_map[ooc_cols] = -1
        else:
            lab_map[i] = j
            j += 1
    df = df.replace({'Label':lab_map})    
    return df.drop(ooc_idx), df[df['Label']==-1]    
def split_data(df, lab_name, ooc_cols=None, oee_cols = None):
    # ooc_cols should be index for lab name! it is because labels in df are already transformed into integer index!

    if ooc_cols is not None:
        print("ooc column: ", lab_name[ooc_cols])
        ooc_idx = df[df['Label']==ooc_cols].index        
    else:
        ooc_idx = pd.Index([])
    if oee_cols is not None:
        print("oee column: ", lab_name[oee_cols])
        oee_idx = df[df['Label']==oee_cols].index        
    else:
        oee_idx = pd.Index([])
    
    
    lab_map = {}
    j=0
    for i,lab in enumerate(lab_name):
        if i == ooc_cols:
            lab_map[ooc_cols] = -1
        elif i== oee_cols:
            lab_map[oee_cols] = -2
        else:
            lab_map[i] = j
            j += 1
    df = df.replace({'Label':lab_map})    
    return df.drop(ooc_idx).drop(oee_idx), df[df['Label']==-1], df[df['Label']==-2]  


def set_multiple_label(df, label_col, label_dic, return_col = False):
    df_y = df[label_col].copy()
    lab_names = df_y.unique()
    drop_idx = pd.Index([])
#     print(label_dic.keys())
    for lab in lab_names:
        if lab in label_dic.keys():
#             print('pass', lab)
            pass
        else:
            print('drop', lab)
            drop_idx = drop_idx.append(df[df[label_col]==lab].index)  
            
    print("original instances: ",len(df))

    print("drop intances: ", len(drop_idx))
    
    df = df.drop(drop_idx)
    
    print("after drop: ",len(df))
#     del df_y, drop_idx    
    if return_col:
        return df_y.replace(label_dic)
    else:        
        return df.replace({label_col:label_dic})
        
        
def apply_clf_with_scaler(clf, scaler, df_x, prob = False):
    df_x_st = scaler.transform(df_x)
    if prob:
        return clf.predict_proba(df_x_st)
    else:
        return clf.predict(df_x_st)
def score_clf_with_scaler(clf, scaler, df_x, df_y):
    df_x_st = scaler.transform(df_x)
    return clf.score(df_x_st, df_y)

import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()