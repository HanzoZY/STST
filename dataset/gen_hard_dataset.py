
import os
import sys

print('Python %s on %s' % (sys.version, sys.platform))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.extend(['../'])
from sklearn.metrics import confusion_matrix
import matplotlib
import pickle
from dataset.video_data import *
matplotlib.use('macosx')

ntu_60_direction = [[8,9],[14,15],[16,17],[18,19],[20,21],[59,60]]
ntu_60_direction_confuse = np.array(ntu_60_direction) - 1
# def ConfMat_NTU(npath, selected_classes=30):
def ConfMat_NTU(npath, train_data_path, train_label_path, val_data_path, val_label_path, selected_classes=30):
    negative_samples = open(npath, 'r').readlines()
    labels = open('../prepare/ntu_60/label.txt', 'r').readlines()
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
    cout_direction_confuse = 0
    classes = labels
    for i in ntu_60_direction_confuse:
        cout_direction_confuse = cout_direction_confuse + confusion[i[0],i[1]] + confusion[i[1],i[0]]
    hard_samples = np.sum(confusion, -1) + np.sum(confusion, 0)
    hard_idx = np.argsort(hard_samples) # asc order as defalut
    hard_idx_selected = set(hard_idx[-selected_classes:])
    trans_dict = {}
    for index, label_idx in enumerate(hard_idx_selected):
        trans_dict[label_idx] = index

    hard_label = np.array(classes)[list(hard_idx_selected)]

    select_samples(hard_idx_selected=hard_idx_selected, trans_dict=trans_dict, data_path=train_data_path,
                   label_path=train_label_path, save_data_path='selected_train_data_joint.npy',
                   save_label_path='selected_train_label.pkl')
    select_samples(hard_idx_selected=hard_idx_selected, trans_dict=trans_dict, data_path=val_data_path,
                   label_path=val_label_path, save_data_path='selected_val_data_joint.npy',
                   save_label_path='selected_val_label.pkl')



def select_samples(hard_idx_selected, trans_dict, data_path, label_path, save_data_path, save_label_path):
    with open(label_path, 'rb') as f:
        sample_name, label = pickle.load(f)
        sample_name = np.array(sample_name)
        label = np.array(label)
        selected_index = np.array([i in hard_idx_selected for i in label])
        selected_index = np.where(selected_index == True)
        selected_lables_origin = label[selected_index]
        selected_lables = [trans_dict[i] for i in selected_lables_origin]
        selected_sample_name = sample_name[selected_index].tolist()
        with open(save_label_path,'wb') as f_w:
            pickle.dump([selected_sample_name,selected_lables], f_w)
        data = np.load(data_path, mmap_mode='r')
        np.save(save_data_path, data[selected_index,:,:,:,:])



if __name__ == '__main__':
    ConfMat_NTU(
        npath='./wrong_path_pre_true.txt')
