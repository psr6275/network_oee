from keras.utils import to_categorical
import matplotlib.pyplot as plt

def apply_nn_clf_with_scaler(clf, scaler, df_x, prob=True):
    df_x_st = scaler.transform(df_x)
    preds = clf.predict(df_x_st)
    if prob:
        return preds
    else:        
        if preds.shape[1]==1:
            return (preds>0.5).astype("int32")
        else:
            return np.argmax(preds, axis=-1)
def score_nn_clf_with_scaler(clf, scaler, df_x, df_y, binary=True):
    df_x_st = scaler.transform(df_x)
    if binary:
        y = df_y.values
    else:
        y = to_categorical(df_y.values)
    score = clf.evaluate(df_x_st,y, batch_size=256)
    return score

def test_nn_accross_dataset(train_df, clf, scaler, lab_map, lab_name1, lab_name2):
    for lab in lab_map:        
        ooc_col = lab
        true_col = lab_map[lab]
        print("OOC Label: ", lab_name1[ooc_col], "/ True Label: ", lab_name2[true_col])
        ooc_idx = (train_df['Label']==ooc_col)
        train_ooc = train_df.loc[ooc_idx]
        train_ooc = train_ooc.loc[:,train_ooc.columns!='Label']
        pred = apply_nn_clf_with_scaler(clf,scaler, train_ooc, False)        
        pred_ht,_ = np.histogram(pred, range=[0,10])        
        plt.bar(np.arange(10),pred_ht)
#         plt.hist(pred)
        plt.title("accuracy: "+str(float(sum(pred ==true_col)/len(pred))))
        plt.xticks(np.arange(10),lab_name2,rotation=90)
        plt.show()