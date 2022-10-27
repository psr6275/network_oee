import numpy as np
import os
import pickle
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

from .skl_utils import set_binary_label,cluster_labels_2017,cluster_labels_2018, set_multiple_label, split_ooc_data

SMALL = 1e-5

class NPsDataSet(TensorDataset):

    def __init__(self, *dataarrays):
        tensors = (torch.tensor(da).clone().detach().float() for da in dataarrays)
        super(NPsDataSet, self).__init__(*tensors)

def load_data_models(dataset, result_dir,lab_name, epochs, ooc_cols, LS, OE):
    cicids_bn = load_cicids_binary_data(dataset,result_dir,True, ooc_cols)
    cicids_m = load_cicids_mult_data(dataset,result_dir,True, ooc_cols)
    
    n_features = cicids_bn[0].shape[1]
    n_hidden = 32
    num_class = len(np.unique(cicids_m[1]))
    
    bn_save_model, mul_save_model = load_model_path(dataset, lab_name, epochs, ooc_cols, LS, OE)

    clf_bn = BinaryNN(n_features, n_hidden)
    clf_mul = MultNN(n_features, n_hidden, num_class)
    
    print("load bn clf: ",bn_save_model)
    print("load mul clf: ",mul_save_model)
    clf_bn.load_state_dict(torch.load(os.path.join(result_dir, bn_save_model),map_location='cpu'))
    clf_mul.load_state_dict(torch.load(os.path.join(result_dir, mul_save_model),map_location='cpu'))
    
    return cicids_bn, cicids_m, clf_bn, clf_mul   

def load_oee_data_models(dataset, result_dir, lab_name, epochs, ooc_cols, oee_cols, LS, VAN=False):
    cicids_m = load_cicids_OEE_mult_data(dataset,result_dir, True, ooc_cols, oee_cols)
    mul_save_model = load_oee_model_path(dataset, lab_name, epochs, ooc_cols, oee_cols, LS, VAN)

    n_features = cicids_m[0].shape[1]
    n_class = len(np.unique(cicids_m[1]))
    n_hidden = 32
    clf_mul = MultNN(n_features, n_hidden, n_class)
    clf_mul.load_state_dict(torch.load(os.path.join(result_dir, mul_save_model),map_location='cpu'))

    return cicids_m, clf_mul


def load_all_data(result_dir = '../results/ids-dataset', ooc_cols = None):
    cicids17_bn = load_cicids_binary_data("2017",result_dir,True, ooc_cols)
    cicids18_bn = load_cicids_binary_data("2018",result_dir,True, ooc_cols)
    cicids17_m = load_cicids_mult_data("2017",result_dir,True, ooc_cols)
    cicids18_m = load_cicids_mult_data("2018",result_dir,True, ooc_cols)
    return cicids17_bn, cicids18_bn, cicids17_m, cicids18_m

def load_model_path(dataset, lab_name, epochs, ooc_cols, LS, OE):   
    if ooc_cols is None:
        oocn = "_"
    else:
        oocn = "_ooc_%s_"%lab_name[ooc_cols]
    if LS:
        lsn = "_LS_"
    else:
        lsn = "_"
    if OE:
        oen = "_OE_"
    else:
        oen = ""
    bn_save_model = "cicids%s%sepochs_%d_bn_clf.pth"%(dataset,oocn,epochs)
    mul_save_model = "cicids%s%sepochs_%d%s%smul_clf.pth"%(dataset,oocn,epochs,lsn,oen)
    return bn_save_model, mul_save_model

def load_oee_model_path(dataset, lab_name, epochs, ooc_cols, oee_cols, LS, van = False):   
    if ooc_cols is None:
        oocn = "_"
    else:
        oocn = "_ooc_%s"%lab_name[ooc_cols]
    if LS:
        lsn = "_LS_"
    else:
        lsn = "_"
    if van:
        oen = "oee_%s_vanilla_"%lab_name[oee_cols]
    else:
        oen = "oee_%s_"%lab_name[oee_cols]
        
    mul_save_model = "cicids%s_epochs_%d%s%s%smul_clf.pth"%(dataset,epochs,oocn,lsn,oen)
    return mul_save_model

def load_oee_scaler_path(dataname, lab_name, oee_cols, ooc_cols=None):
    if ooc_cols is None:
        st_save_path = "cicids%s_oee_%s_stscaler.pkl"%(dataname, lab_name[oee_cols])
    else:
        st_save_path = "cicids%s_ooc_%s_oee_%s_stscaler.pkl"%(dataname,lab_name[ooc_cols], lab_name[oee_cols])
    
    return st_save_path

def load_cicids_binary_data(dataname, lab_cluster, lab_name, result_dir = '../results/ids-dataset', 
                            return_scaler = True, ooc_cols = None):   
    
    assert dataname in ["2017","2018"]
    train_df = pd.read_parquet(os.path.join(result_dir,"cicids%s_traindf.parquet"%dataname))
    test_df = pd.read_parquet(os.path.join(result_dir,"cicids%s_testdf.parquet"%dataname))
    
    # if dataname=="2017":
    #     lab_cluster, lab_name = cluster_labels_2017(True)
    # else:
    #     lab_cluster, lab_name = cluster_labels_2018(True)
    
    train_df_bn = set_binary_label(train_df,label_col='Label', default_class='Benign',return_col=False, ooc_cols=ooc_cols, lab_cluster = lab_cluster)
    test_df_bn = set_binary_label(test_df,label_col='Label', default_class='Benign',return_col=False, ooc_cols=ooc_cols, lab_cluster = lab_cluster)
    

    trainx = train_df_bn.loc[:,train_df.columns!='Label']
    trainy = train_df_bn['Label']
    testx = test_df_bn.loc[:,test_df.columns!='Label']
    testy = test_df_bn['Label'] 
    
    if ooc_cols is None:
        try:
            with open(os.path.join(result_dir,"cicids%s_stscaler_binary.pkl"%dataname),'rb') as f:
                st_scaler_bn = pickle.load(f)
        except:
            st_scaler_bn = StandardScaler()
            st_scaler_bn.fit(trainx)
            with open(os.path.join(result_dir,"cicids%s_stscaler_binary.pkl"%dataname),'wb') as f:
                pickle.dump(st_scaler_bn, f)
    else:
        try:
            with open(os.path.join(result_dir,"cicids%s_ooc_%s_stscaler_binary.pkl"%(dataname,lab_name[ooc_cols])),'rb') as f:
                st_scaler_bn = pickle.load(f)
        except:
            st_scaler_bn = StandardScaler()
            st_scaler_bn.fit(trainx)
            with open(os.path.join(result_dir,"cicids%s_ooc_%s_stscaler_binary.pkl"%(dataname,lab_name[ooc_cols])),'wb') as f:
                pickle.dump(st_scaler_bn, f)

    if return_scaler:
        return trainx, trainy, testx, testy, st_scaler_bn
    else:
        trainx_st = st_scaler_bn.transform(trainx)
        testx_st = st_scaler_bn.transform(testx)
        return trainx_st, trainy, testx_st, testy

def load_cicids_mult_df(dataname, result_dir = '../results/ids-dataset', include_benign=False):
    assert dataname in ["2017","2018"]
    train_df = pd.read_parquet(os.path.join(result_dir,"cicids%s_traindf.parquet"%dataname))
    test_df = pd.read_parquet(os.path.join(result_dir,"cicids%s_testdf.parquet"%dataname))
    
    if dataname=="2017":
        lab_dic, lab_name = cluster_labels_2017(include_benign=include_benign)
    else:
        lab_dic, lab_name = cluster_labels_2018(include_benign=include_benign)    
    
    train_mul= set_multiple_label(train_df,label_col='Label', label_dic = lab_dic, return_col=False)
    test_mul = set_multiple_label(test_df,label_col='Label', label_dic = lab_dic, return_col=False)

    return train_mul, test_mul, lab_dic, lab_name

def load_cicids_OEE_mult_data(dataname="2017",result_dir = '../results/ids-dataset', 
                                return_scaler = True, ooc_cols = None, oee_cols = 0):

    from .skl_utils import split_data

    assert(ooc_cols != oee_cols)
    
    train_mul, test_mul, lab_dic, lab_name = load_cicids_mult_df(dataname, result_dir, True)
    
    
    train_mul, train_ooc, train_oee = split_data(train_mul, lab_name, ooc_cols, oee_cols)
    test_mul, _, test_oee = split_data(test_mul, lab_name, ooc_cols, oee_cols)
    

    trainx_m = train_mul.loc[:,train_mul.columns!='Label']
    trainy_m = train_mul['Label']
    testx_m = test_mul.loc[:,test_mul.columns!='Label']
    testy_m = test_mul['Label']
    trainx_oee = train_oee.loc[:,train_oee.columns!='Label']
    testx_oee = test_oee.loc[:,test_oee.columns!='Label']

    num_class = len(np.unique(trainy_m))
    st_scaler_path = load_oee_scaler_path(dataname, lab_name, oee_cols, ooc_cols)
    print(st_scaler_path)

    try:
        with open(os.path.join(result_dir,st_scaler_path),'rb') as f:
            st_scaler_mul = pickle.load(f)
    except:
        st_scaler_mul = StandardScaler()
        st_scaler_mul.fit(trainx_m)
        with open(os.path.join(result_dir,st_scaler_path),'wb') as f:
            pickle.dump(st_scaler_mul, f)
    
    if return_scaler:
        if ooc_cols is None:
            return trainx_m, trainy_m, testx_m, testy_m, trainx_oee, testx_oee, st_scaler_mul
        else:
            trainx_ooc = train_ooc.loc[:,train_ooc.columns!='Label']
            return trainx_m, trainy_m, testx_m, testy_m, trainx_oee, testx_oee, trainx_ooc, st_scaler_mul
    else:
        trainx_m_st = st_scaler_mul.transform(trainx_m)
        testx_m_st = st_scaler_mul.transform(testx_m)
        trainx_oee_st = st_scaler_mul.transform(trainx_oee)
        testx_oee_st = st_scaler_mul.transform(testx_oee)
        if ooc_cols is None:
            return trainx_m_st, trainy_m, testx_m_st, testy_m, trainx_oee_st, testx_oee_st
        else:            
            trainx_ooc = train_ooc.loc[:,train_ooc.columns!='Label']
            trainx_ooc_st = st_scaler_mul.transform(trainx_ooc)            
            return trainx_m_st, trainy_m, testx_m_st, testy_m, trainx_oee_st, testx_oee_st, train_ooc_st       
    
def load_cicids_mult_data(dataname, lab_dic, lab_name,result_dir = '../results/ids-dataset', 
                            return_scaler = True, ooc_cols = None):   
    
    assert dataname in ["2017","2018"]
    train_df = pd.read_parquet(os.path.join(result_dir,"cicids%s_traindf.parquet"%dataname))
    test_df = pd.read_parquet(os.path.join(result_dir,"cicids%s_testdf.parquet"%dataname))
    
    # if dataname=="2017":
    #     lab_dic, lab_name = cluster_labels_2017()
    # else:
    #     lab_dic, lab_name = cluster_labels_2018()
    
    
    train_mul= set_multiple_label(train_df,label_col='Label', label_dic = lab_dic, return_col=False)
    test_mul = set_multiple_label(test_df,label_col='Label', label_dic = lab_dic, return_col=False)
    
    if ooc_cols is not None:
        train_mul, train_ooc = split_ooc_data(train_mul, lab_name, ooc_cols)
        test_mul, _ = split_ooc_data(test_mul, lab_name, ooc_cols)

    trainx_m = train_mul.loc[:,train_mul.columns!='Label']
    trainy_m = train_mul['Label']
    testx_m = test_mul.loc[:,test_mul.columns!='Label']
    testy_m = test_mul['Label']
    num_class = len(np.unique(testy_m))
    
    if ooc_cols is None:
        try:
            with open(os.path.join(result_dir,"cicids%s_stscaler.pkl"%dataname),'rb') as f:
                st_scaler_mul = pickle.load(f)
        except:
            st_scaler_mul = StandardScaler()
            st_scaler_mul.fit(trainx_m)
            with open(os.path.join(result_dir,"cicids%s_stscaler.pkl"%dataname),'wb') as f:
                pickle.dump(st_scaler_mul, f)
    else:
        trainx_ooc = train_ooc.loc[:,train_ooc.columns!='Label']
        try:
            with open(os.path.join(result_dir,"cicids%s_ooc_%s_stscaler.pkl"%(dataname,lab_name[ooc_cols])),'rb') as f:
                st_scaler_mul = pickle.load(f)
        except:
            st_scaler_mul = StandardScaler()
            st_scaler_mul.fit(trainx_m)
            with open(os.path.join(result_dir,"cicids%s_ooc_%s_stscaler.pkl"%(dataname,lab_name[ooc_cols])),'wb') as f:
                pickle.dump(st_scaler_mul, f)

    if return_scaler:
        if ooc_cols is None:
            return trainx_m, trainy_m, testx_m, testy_m, st_scaler_mul
        else:
            return trainx_m, trainy_m, testx_m, testy_m, trainx_ooc, st_scaler_mul
    else:
        trainx_m_st = st_scaler_mul.transform(trainx_m)
        testx_m_st = st_scaler_mul.transform(testx_m)
        if ooc_cols is None:
            return trainx_m_st, trainy_m, testx_m_st, testy_m
        else:            
            trainx_ooc_st = st_scaler_mul.transform(trainx_ooc)
            return trainx_m_st, trainy_m, testx_m_st, testy_m, train_ooc_st

def dataloader_with_scaler(datadfs, scaler, batch_size, shuffle):
    scaled_df = scaler.transform(datadfs[0])
    data_loader = make_dataloader(scaled_df, datadfs[1].to_numpy(), batch_size = batch_size, shuffle=shuffle)
    return data_loader
    
def make_dataloader(x,y, batch_size, shuffle, scaler = None):
    if len(y.shape)==1:
        y = np.expand_dims(y,axis=1)
    if scaler is not None:
        x = scaler.transform(x)
    return DataLoader(NPsDataSet(x,y), batch_size = batch_size, shuffle=shuffle)

def train_model(clf, train_loader, optimizer, device, loss_clf, epochs, pred_prob=False, test_loader = None, 
                    save_dir = "../results",save_model="cifar_clf.pth", binary=False):
    """
        train network
    """
    if binary:
        accuracy = accuracy_b
    else:
        accuracy = accuracy_m
        
    clf.to(device)
    
    best_acc = 0.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for i,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            if binary==False:
                y = y.long().flatten()
            clf.zero_grad()
            out = clf(x)
            if pred_prob:
                out = torch.log(torch.clamp(out,min=SMALL))
            loss = loss_clf(out,y)
            loss.backward()
            optimizer.step()
            acc = accuracy(out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss
            torch.cuda.empty_cache()
        
        if test_loader:
            _,acc_te,acc_std_te = test_model(clf,test_loader,loss_clf,device,best_acc,save_dir,save_model,pred_prob, binary)
            if best_acc<acc_te:
                best_acc = acc_te    
            logging.info(
            # print(
                    "\nEpoch [{}/{}]\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 train {:.3f} ({:.3f})\t"
                    "Prec@1 test {:.3f} ({:.3f})   \t".format(
                        (epoch+1),
                        epochs,
                        losses.avg.detach().cpu(),
                        losses.std.detach().cpu(),
                        accs.avg.detach().cpu(),
                        accs.std.detach().cpu(),
                        acc_te,
                        acc_std_te,                   
                    )
            )
        else:
            torch.save(clf.state_dict(),os.path.join(save_dir, save_model))
            logging.info(
            # print(
                "\nEpoch [{}/{}]\t"
                "Loss {:.4f} ({:.4f})\t"
                "Prec@1 train {:.3f} ({:.3f})   \t".format(
                    (epoch+1),
                    epochs,
                    losses.avg.detach().cpu(),
                    losses.std.detach().cpu(),
                    accs.avg.detach().cpu(),
                    accs.std.detach().cpu(),             
                )
            )
    clf.cpu()
    return clf

def train_model_with_oe_KL(clf, train_loader, outlier_loader, num_classes, optimizer, device, 
                          loss_in, loss_out, weight_out, epochs,pred_prob = False, test_loader = None,
                          save_dir = '../results', save_model="cifar_clf.pth", binary=False):
    clf.to(device)
    best_acc = 0.0
    if binary:
        accuracy = accuracy_b
    else:
        accuracy = accuracy_m
    
    Ndata = len(train_loader.dataset.tensors[0])
    Odata = len(outlier_loader.dataset.tensors[0])
    outds = outlier_loader.dataset
    
    for epoch in range(epochs):
        losses1 = AverageVarMeter()
        losses2 = AverageVarMeter()
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        
        clf.train()
        
        if Odata>=Ndata:
            outset, _ = torch.utils.data.random_split(outds, [Ndata, Odata-Ndata])
            outlier_loader_ = DataLoader(outset, batch_size=train_loader.batch_size, shuffle=True)        
        else:
            outlier_loader_ = outlier_loader
        
        outloader = iter(outlier_loader_)
        
        for i,in_set in enumerate(train_loader):
            x,y = in_set[0].to(device),in_set[1].to(device).long().flatten()
            
            try: 
                out_set = outloader.next()
            except StopIteration:
                outloader = iter(outlier_loader_)
                out_set = outloader.next()

            x_out = out_set[0].to(device)
            
            clf.zero_grad()
            pred_in = clf(x)
            pred_out = clf(x_out)
            
            
#             pred_out = pred_out.softmax(dim=1)

            loss1 = loss_in(pred_in,y)
            
                
            loss2 = weight_out*loss_out(torch.log_softmax(pred_out,dim=1), torch.ones_like(pred_out)/num_classes)
            loss = loss1+loss2
            
            loss.backward()
            optimizer.step()
            
            acc = accuracy(pred_in.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            losses1.update(loss1.detach().cpu(),x.size(0))
            losses2.update(loss2.detach().cpu(),x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, pred_in,x,y,loss, pred_out, x_out
            torch.cuda.empty_cache()

        if test_loader:
            _,acc_te,acc_std_te = test_model(clf,test_loader,loss_in,device,best_acc,save_dir,save_model,pred_prob,binary)
            if best_acc<acc_te:
                best_acc = acc_te    
            logging.info(
            # print(
                    "\nEpoch [{}/{}]\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Loss1/Loss2 {:.4f}/{:.4f}\t"
                    "Prec@1 train {:.3f} ({:.3f})\t"
                    "Prec@1 test {:.3f} ({:.3f})   \t".format(
                        (epoch+1),
                        epochs,
                        losses.avg.detach().cpu(),
                        losses.std.detach().cpu(),
                        losses1.avg.detach().cpu(),
                        losses2.avg.detach().cpu(),
                        accs.avg.detach().cpu(),
                        accs.std.detach().cpu(),
                        acc_te,
                        acc_std_te,                   
                    )
            )
        else:
            torch.save(clf.state_dict(),os.path.join(save_dir, save_model))
            logging.info(
            # print(
                "\nEpoch [{}/{}]\t"
                "Loss {:.4f} ({:.4f})\t"
                "Loss1/Loss2 {:.4f}/{:.4f}\t"
                "Prec@1 train {:.3f} ({:.3f})   \t".format(
                    (epoch+1),
                    epochs,
                    losses.avg.detach().cpu(),
                    losses.std.detach().cpu(),
                    losses1.avg.detach().cpu(),
                    losses2.avg.detach().cpu(),
                    accs.avg.detach().cpu(),
                    accs.std.detach().cpu(),             
                )
            )
    clf.cpu()
    
    return clf

def test_model(model, test_loader, criterion, device, best_acc=0.0,save_dir = "../results/",save_model = "ckpt.pth",pred_prob = False, binary=False):
    model.eval().to(device)
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    
    if binary:
        accuracy= accuracy_b
    else:
        accuracy= accuracy_m
        
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(device)
            if binary:
                y = y.to(device).view(-1,1)
            else:
                y = y.to(device).long().flatten()

            p_y = model(x)
            if pred_prob:
                p_y = torch.log(torch.clamp(p_y,min=SMALL))
             
            loss = criterion(p_y,y)

            acc = accuracy(p_y.detach().cpu(), y.detach().cpu())

            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del x,y,p_y, loss, acc
            torch.cuda.empty_cache()
    #         print(acc)
        if accs.avg>best_acc:
            torch.save(model.state_dict(),os.path.join(save_dir, save_model))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu(),accs.std.detach().cpu()

class AverageVarMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.sum2 = 0
        self.std = 0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val = val
#         print(val)
        self.sum2 += (val**2)*n
        self.sum +=val*n
        self.count +=n
        self.avg = self.sum / self.count
        self.std = torch.sqrt(self.sum2/self.count-self.avg**2)

def accuracy_m(output, target, topk=(1,)):
    '''Compute the top1 and top k accuracy'''
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res

def accuracy_b(output, target, thres = 0.5):
    '''Compute the binary accuracy'''
#     print(output.ndim, target.size(), output.size())
#     assert target.size() == output.size()
#     if output.ndim == 1:
    y_prob = output>thres 
    return [torch.tensor((target == y_prob).sum().item()*100 / target.size(0))]
    

class BinaryNN(nn.Module):
    def __init__(self, n_features,n_hidden=32):
        super(BinaryNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

class MultNN(nn.Module):
    def __init__(self, n_features,n_hidden=32, num_class=10):
        super(MultNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, num_class),
        )

    def forward(self, x):
        return self.network(x)
    
def get_prediction(model, testloader, device):
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for data in testloader:
            preds.append(model(data[0].to(device)).detach().cpu())
        preds = torch.cat(preds,dim=0)
    return preds
    
import copy

class HClassifier:
    def __init__(self, model_bn, model_mul, device, thres_bn=0.5, thres_mul=0.2, ooc_class = 7):
        self.model_bn = model_bn
        self.model_mul = model_mul
        self.thres_bn = thres_bn
        self.thres_mul = thres_mul
        self.ooc_class = 7
        self.device = device
    def predict(self, bnloader, mulloader, thres_bn=None, thres_mul=None):
        if thres_bn is None:
            thres_bn = self.thres_bn        
        if thres_mul is None:
            thres_mul = self.thres_mul
        tot_pred = get_hier_prediction(self.model_bn, self.model_mul, bnloader, mulloader, self.device,\
                            thres_bn, thres_mul, self.ooc_class)    
        return tot_pred

def get_hier_prediction(model_bn, model_mul, bnloader, mulloader, device, 
                        thres_bn = 0.5, thres_mul = 0.2, ooc = 10):
    preds_bn = get_prediction(model_bn, bnloader, device)
    # ya_idx = preds_bn.flatten()>=thres_bn
    yb_idx = preds_bn.flatten()<thres_bn

    pred_mul = get_prediction(model_mul, mulloader, device)
    pred_max,pred_cls = torch.max(torch.softmax(pred_mul, dim=1), dim=1)
    idx_conf = pred_max.flatten()>=thres_mul
    idx_nconf = torch.logical_not(idx_conf)
    # print(sum(idx_nconf))

    tot_pred = copy.deepcopy(pred_cls.flatten())
    tot_pred[yb_idx*idx_nconf] = -1.0
    tot_pred[yb_idx*idx_conf] = 10.0

    return tot_pred