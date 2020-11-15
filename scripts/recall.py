# from scripts.recall_fpr_fdr import get_recall_fpr
# import sys
import numpy as np
from sklearn.metrics import confusion_matrix
# import time


def binary_search(arr, pos_left, pos_right, req, count):
    if pos_right >= pos_left:
        mid = pos_left + (pos_right-1)//2
        curr_pred = np.ones(shape=len(arr))
        curr_pred[mid+1:] = 0.
        tn, fp, fn, tp = confusion_matrix(arr, curr_pred).ravel()
        # curr_tpr = tp / (tp + fn)
        curr_fpr = fp / (fp + tn)

        count += 1
        if curr_fpr == req:
            return mid
        elif curr_fpr > req:
            return binary_search(arr, pos_left, mid-1, req, count)
        else:
            return mid
    else:
        return pos_left


def get_fast_recall_fpr(y, pred, req_fpr=0.01):
    num_examples = len(y)
    data = np.zeros(shape=(num_examples,2))
    data[:,0] = pred
    data[:,1] = y

    rev_sorted_data = data[np.argsort(-data[:,0]), :]

    possible_index = binary_search(rev_sorted_data[:, 1], 0, num_examples, req_fpr, 1)

    left_optimistic = max(possible_index-50, 0)
    right_optimistic = min(possible_index+50, num_examples)

    for i in range(left_optimistic, right_optimistic):
        curr_pred = np.ones(shape=num_examples)
        curr_pred[i + 1:] = 0.
        tn, fp, fn, tp = confusion_matrix(rev_sorted_data[:, 1], curr_pred).ravel()
        curr_tpr = tp / (tp + fn)
        curr_fpr = fp / (fp + tn)

        if curr_fpr > req_fpr:
            break

    return rev_sorted_data[i, 0], curr_tpr, curr_fpr

# if len(sys.argv) < 3:
#     print('python file.py predfile req_fpr')
#     exit(1)
#
# d = np.loadtxt(sys.argv[1])
# # print(d)
# # exit(1)
# req_fpr = float(sys.argv[2])
#
# # print('fast')
# # start = time.time()
# item, tpr, fpr = get_fast_recall_fpr(d[:, 0], d[:, 1], req_fpr)
# stop = time.time()
# print('curr time:', stop-start)
# print('req_fpr:', req_fpr, 'item:', item, 'tpr:', tpr, 'fpr:', fpr)
#
# # verify
# p1 = d[:,1].copy()
# p1[p1>=item] = 1.
# p1[p1<item] = 0.
# tn, fp, fn, tp = confusion_matrix(d[:,0], p1).ravel()
# curr_tpr = tp / (tp + fn)
# curr_fpr = fp / (fp + tn)
# print('check tpr:', curr_tpr,
#       'check fpr:', curr_fpr)
# # exit(1)
#
#
# start = time.time()
# item, tpr, fpr = get_recall_fpr(d[:, 0], d[:, 1], req_fpr)
# stop = time.time()
# print('prev time:', stop - start)
# print('req_fpr:', req_fpr, 'item:', item, 'tpr:', tpr, 'fpr:', fpr)
#
# # verify
# p1 = 0
# print('p1:', p1)
# p1 = d[:,1]
# p1[p1>=item] = 1.
# p1[p1<item] = 0.
# tn, fp, fn, tp = confusion_matrix(d[:,0], p1).ravel()
# curr_tpr = tp / (tp + fn)
# curr_fpr = fp / (fp + tn)
# print('check tpr:', curr_tpr,
#       'check fpr:', curr_fpr)
# exit(1)


