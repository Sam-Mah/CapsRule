import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, precision_score, recall_score, f1_score
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
N_CLASSSES = 4
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def confusion_metrics_tf(actual_classes, model, session, feed_dict):
    actuals = tf.argmax(actual_classes, 1)
    predictions = tf.argmax(model, 1)
    actuals = session.run(actuals, feed_dict)
    predictions = session.run(predictions, feed_dict)

    lbls = [*range(N_CLASSSES)]
    mcm = multilabel_confusion_matrix(actuals, predictions, labels=lbls)
    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]

    # print(calculate_measures(np.mean(tp), np.mean(tn), np.mean(fp), np.mean(fn), 0))
    #true label being i-th (row) class and the predicted label being jth (column)
    cm = confusion_matrix(actuals, predictions, labels=lbls, sample_weight=None)
    return cm, np.mean(tp), np.mean(tn), np.mean(fp), np.mean(fn), actuals, predictions
def confusion_metrics_basic(actuals, predictions):
    lbls = [*range(N_CLASSSES)]
    mcm = multilabel_confusion_matrix(actuals, predictions, labels=lbls)
    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    cm = confusion_matrix(actuals, predictions, labels=lbls, sample_weight=None)
    return cm, np.mean(tp), np.mean(tn), np.mean(fp), np.mean(fn)
def Micro_calculate_measures(tp, tn, fp, fn, uncovered_sample):
    fn += uncovered_sample
    try:
        tpr = float(tp) / (float(tp) + float(fn))
        accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
        recall = tpr
        precision = float(tp) / (float(tp) + float(fp))

        f1_score = (2 * (precision * recall)) / (precision + recall)
        fp_rate = float(fp) / (float(fp) + float(tn))
        fn_rate = float(fn) / (float(fn) + float(tp))

        # return precision, recall, f1_score, accuracy, fp_rate, fn_rate
        PR = str(round(precision * 100, 2))
        RC = str(round(recall * 100, 2))
        F1 = str(round(f1_score * 100, 2))
        ACC = str(round(accuracy * 100, 2))
        FPR = str(round(fp_rate * 100, 2))
        FNR = str(round(fn_rate * 100, 2))

        data_pd = [['PR', PR], ['RC', RC], ['F1', F1], ['ACC', ACC], ['FPR', FPR], ['FNR', FNR], ['tp', tp], ['tn', tn],
                   ['fp', fp], ['fn', fn]]

        df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])

    except Exception as e:
        print(e)
        data_pd = [['PR', 'Err'], ['RC', 'Err'], ['F1', 'Err'], ['ACC', 'Err'], ['FPR', 'Err'], ['FNR', 'Err']]

        df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])
    return df
def Macro_calculate_measures_tf(y_true, y_pred, session, feed_dict):
    actuals = tf.argmax(y_true, 1)
    predictions = tf.argmax(y_pred, 1)
    y_true = session.run(actuals, feed_dict)
    y_pred = session.run(predictions, feed_dict)
    pr=  precision_score(y_true, y_pred, average='macro')
    rc = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return pr, rc, f1

def Macro_calculate_measures_basic(y_true, y_pred):
    pr = precision_score(y_true, y_pred, average='macro')
    rc = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return pr, rc, f1
