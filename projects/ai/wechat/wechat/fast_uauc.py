import os
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata
from joblib import Parallel, delayed

@njit
def _auc(actual, pred_ranks):
    actual = np.asarray(actual)
    pred_ranks = np.asarray(pred_ranks)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def auc(actual, predicted):
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)

def uAUC(y_true, y_pred, userids, weights):
    num_labels = y_pred.shape[1]

    def uAUC_infunc(i):
        uauc_df = pd.DataFrame()
        uauc_df['userid'] = userids
        uauc_df['y_true'] = y_true[:, i]
        uauc_df['y_pred'] = y_pred[:, i]

        label_nunique = uauc_df.groupby(by='userid')['y_true'].transform('nunique')
        uauc_df = uauc_df[label_nunique == 2]
        
        aucs = uauc_df.groupby(by='userid').apply(
            lambda x: auc(x['y_true'].values, x['y_pred'].values))
        return np.mean(aucs)

    ## Parallel will hang on v100 machine, _pickle.PicklingError: Could not pickle the task to send it to the workers.
    # if 'tione' in os.environ['PATH']: 
    uauc = np.asarray([uAUC_infunc(i) for i in range(num_labels)])
    # else:
    #   uauc = Parallel(n_jobs=len(weights))(delayed(uAUC_infunc)(i) for i in range(num_labels))

    return np.average(uauc, weights=weights), uauc
  