# original code is from https://github.com/ShiyuLiang/odin-pytorch/blob/master/code/calMetric.py
# Modified by Kimin Lee

import numpy as np


def tpr95(dir_name, task='OOD'):
    # calculate the falsepositive error when tpr is 95%
    if task == 'OOD':
        cifar = np.loadtxt('%s/confidence_Base_In.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' % dir_name, delimiter=',')
    elif task == 'mis':
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' % dir_name, delimiter=',')
    else:
        raise Exception('unknown task.')

    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start) / 200000  # precision:200000

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.96 and tpr >= 0.94:
            fpr += error2
            total += 1
    if total == 0:
        print('corner case')
        fprBase = 1
    else:
        fprBase = fpr / total

    return fprBase


def auroc(dir_name, task='OOD'):
    # calculate the AUROC
    if task == 'OOD':
        f1 = open('%s/Update_Base_ROC_tpr.txt' % dir_name, 'w')
        f2 = open('%s/Update_Base_ROC_fpr.txt' % dir_name, 'w')
        cifar = np.loadtxt('%s/confidence_Base_In.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' % dir_name, delimiter=',')
    elif task == 'mis':
        f1 = open('%s/Update_Base_ROC_tpr_mis.txt' % dir_name, 'w')
        f2 = open('%s/Update_Base_ROC_fpr_mis.txt' % dir_name, 'w')
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' % dir_name, delimiter=',')
    else:
        raise Exception('unknown task.')

    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start) / 200000

    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        f1.write("{}\n".format(tpr))
        f2.write("{}\n".format(fpr))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    f1.close()
    f2.close()
    return aurocBase


def auprIn(dir_name, task='OOD'):
    # calculate the AUPR
    if task == 'OOD':
        cifar = np.loadtxt('%s/confidence_Base_In.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' % dir_name, delimiter=',')
    elif task == 'mis':
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' % dir_name, delimiter=',')

    precisionVec = []
    recallVec = []
    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start) / 200000

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta))  # / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta))  # / np.float(len(Y1))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def auprOut(dir_name, task='OOD'):
    # calculate the AUPR
    if task == 'OOD':
        cifar = np.loadtxt('%s/confidence_Base_In.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' % dir_name, delimiter=',')
    elif task == 'mis':
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' % dir_name, delimiter=',')
    else:
        raise Exception('unknown task.')

    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start) / 200000

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta))  # / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta))  # / np.float(len(Y1))
        if tp + fp == 0:
            break
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1))
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def detection(dir_name, task='OOD'):
    # calculate the minimum detection error
    if task == 'OOD':
        cifar = np.loadtxt('%s/confidence_Base_In.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Out.txt' % dir_name, delimiter=',')
    elif task == 'mis':
        cifar = np.loadtxt('%s/confidence_Base_Succ.txt' % dir_name, delimiter=',')
        other = np.loadtxt('%s/confidence_Base_Err.txt' % dir_name, delimiter=',')
    else:
        raise Exception('unknown task.')

    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start) / 200000

    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    return errorBase


def metric(dir_name, task):
    print("{}{:>34}".format(task, "Performance of Baseline detector"))
    fprBase = tpr95(dir_name, task)
    print("{:20}{:13.3f}%".format("TNR at TPR 95%:", (1 - fprBase) * 100))
    aurocBase = auroc(dir_name, task)
    print("{:20}{:13.3f}%".format("AUROC:", aurocBase * 100))
    errorBase = detection(dir_name, task)
    print("{:20}{:13.3f}%".format("Detection acc:", (1 - errorBase) * 100))
    auprinBase = auprIn(dir_name, task)
    print("{:20}{:13.3f}%".format("AUPR In:", auprinBase * 100))
    auproutBase = auprOut(dir_name, task)
    print("{:20}{:13.3f}%".format("AUPR Out:", auproutBase * 100))
