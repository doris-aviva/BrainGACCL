import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix


# metrics
def get_acc(pred, label):
    return accuracy_score(label, pred)


def get_sen(pred, label):
    return recall_score(label, pred)


def get_spe(pred, label):
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    return tn / (tn + fp)


def get_pre(pred, label):
    return precision_score(label, pred)


def get_f1(pred, label):
    return f1_score(label, pred)


def get_auc(prob, label):
    score = prob[:, 1]
    return roc_auc_score(label, score)


def get_bac(pred, label):
    return balanced_accuracy_score(label, pred)


def evaluate(pred, prob, label):
    acc = get_acc(pred, label)
    sen = get_sen(pred, label)
    spe = get_spe(pred, label)
    pre = get_pre(pred, label)
    f1 = get_f1(pred, label)
    auc = get_auc(prob, label)
    bac = get_bac(pred, label)
    return acc, sen, spe, pre, f1, auc, bac