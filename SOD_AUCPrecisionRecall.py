from label_evaluation import label_evaluation
import numpy as np
from sklearn import metrics


def score2label_threshold(score, percentage=0.95):
    import math
    score = np.array(score)
    temp_array = score.copy()
    temp_array.sort()
    if percentage == 0:
        threshold = temp_array[0]
    else:
        threshold = temp_array[math.floor(len(temp_array) * percentage)-1]

    label = np.zeros(len(score), dtype=int)
    for ii, s in enumerate(score):
        if s > threshold:
            label[ii] = 1
    return label

def get_preformance(score, y_true):
    auc_roc = metrics.roc_auc_score(y_true, score)
    precision, recall, _ = metrics.precision_recall_curve(y_true, score)
    auc_pr = metrics.auc(recall, precision)
    return auc_roc, auc_pr



def evaluate(name, isscore, value, y_true, threshold=0.99, txt=''):
    '''
    :param isscore: True:value=score, False:value=label
    :param value: score or label
    '''
    if isscore:
        score = np.copy(value)
        roc, pr = get_preformance(score, y_true)
        y_pre = score2label_threshold(score, threshold)
    else:
        y_pre = np.copy(value)
        roc = 0
        pr = 0

    if sum(y_true) == 0:
        text = txt + 'In this section, y_true is all 0!!\n'
        return 0, 0, 0, 0, 0, 0, 0, 0, y_pre, text

    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pre,
                                                               average="binary")
    newresult = label_evaluation(y_true, y_pre)
    precision_eval, recall_eval, f1_eval, _ = precision_recall_fscore_support(y_true=y_true, y_pred=newresult,
                                                                              average="binary")
    text = str(name) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' + str(roc) + ',' + str(
        pr) + ',' + str(precision_eval) + ',' + str(recall_eval) + ',' + str(f1_eval) + ',' + txt + '\n'
    return precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval, newresult, text
