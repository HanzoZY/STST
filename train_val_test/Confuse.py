from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('macosx')
negative_samples = open('./wrong_path_pre_true.txt', 'r').readlines()
labels = open('../prepare/ntu_60/label.txt', 'r').readlines()
num_neg = len(negative_samples)
pred = np.array([])
tgt = np.array([])
for i in range(len(negative_samples)):
    sample_line = negative_samples[i].strip('\n')
    samples = sample_line.split(',')
    pred_item = int(samples[1])
    tgt_item = int(samples[2])
    pred = np.append(pred, pred_item)
    tgt = np.append(tgt, tgt_item)

for i in range(len(labels)):
    labels[i] = labels[i].strip('\n')
confusion = confusion_matrix(pred, tgt)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
classes = labels
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('pred')
plt.ylabel('tgt')
plt.xticks(rotation=90)
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index])

plt.show()