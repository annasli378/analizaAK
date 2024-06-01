import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# Functions
def read_fold(path_to_data_train, path_to_data_test, fold_k):
    # read data from each folder
    X_train = pd.read_csv(path_to_data_train  + str(fold_k) + '_x_train.csv', header=None).values
    Y_train = pd.read_csv(path_to_data_train  + str(fold_k) + '_y_train.csv', header=None).values - 1
    X_test = pd.read_csv(path_to_data_test  + str(fold_k) + '_x_test.csv').values
    Y_test = pd.read_csv(path_to_data_test  + str(fold_k) + '_y_test.csv', header=None).values - 1
    return X_train, Y_train, X_test, Y_test

def svn_train(DANE_FOLDY, gamma, C,i, plot_acc=False):
    # load data from each fold
    all_preds = np.zeros(test_size)
    all_tru = np.zeros(test_size)
    cnt=0
    for k in range(0,kmax):
        X_train, Y_train, X_test, Y_test = DANE_FOLDY[k][0], DANE_FOLDY[k][1], DANE_FOLDY[k][2],DANE_FOLDY[k][3]
        X_fold = X_train[:, 0:i]
        Y_fold = Y_train
        X_test = X_test[:, 0:i]
        # model SVM
        model = svm.SVC( gamma=gamma, C=C).fit(X_fold, Y_fold.ravel())
        preds_test = model.predict(X_test)
        if plot_acc: print(f'fold: {k}, acc: {metrics.balanced_accuracy_score(Y_test, preds_test)}')
        # all prediction
        for c in range(0, len(preds_test)):
            all_preds[cnt] = preds_test[c]
            all_tru[cnt] = Y_test[c]
            cnt=cnt+1
    acc = metrics.balanced_accuracy_score(all_tru, all_preds)
    return acc, all_tru, all_preds

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def print_confusion_matrix(y_true, y_pred, classifier_name, name_to_save):
    # Confusion matrix for light pink confusion matices
    light_pink = cmap_map(lambda x: x/2 + 0.35, matplotlib.cm.pink)
    y_true = y_true + 1
    y_pred = y_pred + 1
    labels = [1,2,3]
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.title(classifier_name, fontsize = 14)
    ax = sns.heatmap(df_cmx, annot=True, fmt='g', square=True, cmap=light_pink)
    plt.xlabel("Predicted classes", fontsize = 15, labelpad=3)
    plt.ylabel("True classes", fontsize = 16, labelpad=3)
    plt.savefig(path_to_save + name_to_save, transparent=True)
    plt.show()
    plt.close()

    report = classification_report(y_true, y_pred,output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(path_to_save+ 'classification_report_' +  classifier_name +'.csv')
    # Save prediction 
    save_results = pd.DataFrame(y_pred)
    save_results.to_csv(path_to_save+'wyniki_preds_' +  classifier_name +'.csv')
    save_results = pd.DataFrame(y_true)
    save_results.to_csv(path_to_save+'wyniki_true_' +  classifier_name +'.csv')


def iterative_svn(DANE_FOLDY):
  svn_tmp=[]
  for i in range(5,100):
        print(f'{i}')
        for C in [  0.01, 0.05, 0.1, 0.5,  1, 5, 10, 50, 100]:
            for gamma in [  0.01, 0.05, 0.1, 0.5,  1, 5, 10, 50, 100, 'auto', 'scale']:
                accsvn,_,_ = svn_train(DANE_FOLDY, gamma=gamma, C=C,i=i)
                svn_tmp.append({'ft_num': i, 'name':'SVN', 'C': C,'gamma': gamma,  'acc': accsvn})
  return svn_tmp

path_folder = ""
path_to_data_train = path_folder +  "/k_"
path_to_data_test = path_to_data_train # Here may be different test and train paths (different data modifications)
path_to_save = path_folder + "/"
kmax = 54 # number of folds - patients

# Table for data in each fold:
DANE_FOLDY = []
for k in range(0,kmax):
  DANE_FOLDY.append(read_fold(path_to_data_train,path_to_data_test, k+1))

test_size = 0
for k in range(0,kmax):
    test_size += len(DANE_FOLDY[k][3])

svn_tmp=iterative_svn(DANE_FOLDY)

SVN_res = pd.DataFrame(svn_tmp)
maxidx_svn = SVN_res['acc'].idxmax()
print(
    f"SVN acc max: {SVN_res['acc'].iloc[maxidx_svn]}, number of features: {SVN_res['ft_num'].iloc[maxidx_svn]}, C: {SVN_res['C'].iloc[maxidx_svn]}, gamma: {SVN_res['gamma'].iloc[maxidx_svn]}")

# Print pretty pink matrix 
accsvn, sy_true, sy_pred = svn_train(DANE_FOLDY, gamma=SVN_res['gamma'].iloc[maxidx_svn], C=SVN_res['C'].iloc[maxidx_svn],i=SVN_res['ft_num'].iloc[maxidx_svn], plot_acc=True)
print(accsvn)
print_confusion_matrix(sy_true, sy_pred, 'SVN' , 'confusion_matrix_dermo_nn_hfus_SVN.png')
