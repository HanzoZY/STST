import os
import sys

print('Python %s on %s' % (sys.version, sys.platform))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.extend(['../'])

import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from train_val_test import train_val_model, parser_args
from utility.log import TimerBlock, IteratorTimer
from method_choose.data_choose import data_choose, init_seed
from method_choose.model_choose import model_choose

matplotlib.use('macosx')

direction = [[8,9],[14,15],[16,17],[18,19],[20,21],[59,60]]
direction_confuse = np.array(direction) - 1
plt.rc('font',family='Arial Unicode MS')
edge_dict = {
    'shrec_skeleton' : ((0, 1),
        (1, 2), (2, 3), (3, 4), (4, 5),
        (1, 6), (6, 7), (7, 8), (8, 9),
        (1, 10), (10, 11), (11, 12), (12, 13),
        (1, 14), (14, 15), (15, 16), (16, 17),
        (1, 18), (18, 19), (19, 20), (20, 21)),
    'ntu_skeleton' : ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
        (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
        (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)),
    'kinetics_skeleton' : ((4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14))
}

def vis_all(data, edge=None, is_3d=False, tag='', pause=0.01):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('macosx')
    N, C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    # add label
    fig.suptitle(tag)

    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    pose = []
    for m in range(M):
        a = []
        for i in range(len(edge)):
            if is_3d:
                a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
            else:
                a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
        pose.append(a)
    ax.axis([-1, 1, -1, 1])
    if is_3d:
        ax.set_zlim3d(-1, 1)
    plt.axis('on')
    # while True:
    for t in range(T):
        for m in range(M):
            for i, (v1, v2) in enumerate(edge):
                x1 = data[0, :2, t, v1, m]
                x2 = data[0, :2, t, v2, m]
                if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                    pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                    pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                    if is_3d:
                        pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
        fig.canvas.draw()
        if t % 2 == 0 and t <= 58:
            # plt.savefig('./skeleton_sequence/' + str(t) + '.eps', dpi=300,format='eps')
            plt.savefig('./skeleton_sequence/' + str(t) + '.png', dpi=300, format='png')
            # pass
        plt.pause(pause)
    plt.close()
    plt.ioff()






def ConfMat(npath):
    negative_samples = open(npath, 'r').readlines()
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
    cout_direction_confuse = 0
    for i in direction_confuse:
        cout_direction_confuse = cout_direction_confuse + confusion[i[0],i[1]] + confusion[i[1],i[0]]

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

def Statistic(shown_num, npath):
    negative_samples = open(npath, 'r').readlines()
    labels = open('../prepare/ntu_60/label.txt', 'r').readlines()



    num_neg = len(negative_samples)
    pred = np.array([])
    tgt = np.array([])
    for i in range(num_neg):
        sample_line = negative_samples[i].strip('\n')
        samples = sample_line.split(',')
        pred_item = int(samples[1])
        tgt_item = int(samples[2])
        pred = np.append(pred, pred_item)
        tgt = np.append(tgt, tgt_item)

    for i in range(len(labels)):
        labels[i] = labels[i].strip('\n')
    num_class = len(labels)
    confusion = confusion_matrix(pred, tgt)

    flat_conf = confusion.reshape((num_class*num_class))
    idx_mat = np.zeros((num_class,num_class,2),dtype=np.int)
    for i in range(num_class):
        idx_mat[i,:,0] = i
    for j in range(num_class):
        idx_mat[:,j,1] = j
    flat_idx_mat = idx_mat.reshape((num_class*num_class,2))
    label_flat_array = []
    for i in range(flat_idx_mat.shape[0]):
        tag = labels[flat_idx_mat[i,0]] + '/' + labels[flat_idx_mat[i,1]]
        label_flat_array.append(tag)
    label_flat_array = np.array(label_flat_array)
    sorted_idx = np.argsort(flat_conf)[::-1]
    scores = flat_conf[sorted_idx]
    tags = label_flat_array[sorted_idx]
    all_wrong = np.sum(flat_conf)


    scores_pr = scores[:shown_num]

    selected_wrong = np.sum(scores_pr)
    tags_pr = tags[:shown_num]
    plt.xlabel('condition')
    plt.ylabel('amount')
    plt.xticks(rotation=90)
    plt.bar(tags_pr, scores_pr)

    print('all wrong:', all_wrong)
    print('selected wrong', selected_wrong)
    print(selected_wrong/all_wrong)

    plt.show()


def auto_text(rects,width):
    for rect in rects:
        plt.text(rect.get_x()+width, rect.get_height(), int(rect.get_height()), ha='center', va='bottom')

def Statistic_compare(npath_0, npath_1):
    bar_width = 0.35
    x = np.arange(len(direction_confuse))
    negative_samples_0 = open(npath_0, 'r').readlines()
    negative_samples_1 = open(npath_1, 'r').readlines()
    labels = open('../prepare/ntu_60/label.txt', 'r').readlines()
    for i in range(len(labels)):
        labels[i] = labels[i].strip('\n')[:-1]



    num_neg_0 = len(negative_samples_0)
    num_neg_1 = len(negative_samples_1)
    neg_0_count_direction = np.zeros(6)
    neg_1_count_direction = np.zeros(6)
    direction_confuse_label = [labels[i[0]]+'\n and \n '+labels[i[1]] for i in direction_confuse]
    for i in range(num_neg_0):
        sample_line = negative_samples_0[i].strip('\n')
        samples = sample_line.split(',')
        pred_item = int(samples[1])
        tgt_item = int(samples[2])
        for idx, f in enumerate(direction_confuse):
            if pred_item in f and tgt_item in f:
                neg_0_count_direction[idx] += 1
    for i in range(num_neg_1):
        sample_line = negative_samples_1[i].strip('\n')
        samples = sample_line.split(',')
        pred_item = int(samples[1])
        tgt_item = int(samples[2])
        for idx, f in enumerate(direction_confuse):
            if pred_item in f and tgt_item in f:
                neg_1_count_direction[idx] += 1

    rect0 = plt.bar(x, neg_0_count_direction, width=bar_width, color='#8FBC8F',align="center",label="With TTB")
    rect1 = plt.bar(x+bar_width, neg_1_count_direction, width=bar_width, color='#008B8B', align="center", label="With DTTB")
    plt.ylim(0, 140)
    plt.ylabel("amount of confusion")
    plt.xticks(x + bar_width / 2, direction_confuse_label, rotation=60)
    # plt.xticks(rotation=90)
    auto_text(rect0,bar_width / 2)
    auto_text(rect1,bar_width / 2)
    plt.legend()



    plt.show()


def compare_samples(p_path,n_path,n_config):
    p_negative_samples = open(p_path, 'r').readlines()
    n_negative_samples = open(n_path, 'r').readlines()
    labels = open('../prepare/ntu_60/label.txt', 'r').readlines()
    for i in range(len(labels)):
        labels[i] = labels[i].strip('\n')
    p_neg_num = len(p_negative_samples)
    n_neg_num = len(n_negative_samples)
    p_id_dict = {}
    n_id_dict = {}
    p_pred = np.array([],dtype=np.int)
    p_tgt = np.array([],dtype=np.int)
    n_pred = np.array([],dtype=np.int)
    n_tgt = np.array([],dtype=np.int)


    for i in range(p_neg_num):
        sample_line = p_negative_samples[i].strip('\n')
        samples = sample_line.split(',')
        p_id_dict[str(samples[0])]=i
        p_pred_item = int(samples[1])
        p_tgt_item = int(samples[2])
        p_pred = np.append(p_pred, p_pred_item)
        p_tgt = np.append(p_tgt, p_tgt_item)

    for j in range(n_neg_num):
        sample_line = n_negative_samples[j].strip('\n')
        samples = sample_line.split(',')
        n_id_dict[str(samples[0])]=j
        n_pred_item = int(samples[1])
        n_tgt_item = int(samples[2])
        n_pred = np.append(n_pred, n_pred_item)
        n_tgt = np.append(n_tgt, n_tgt_item)

    good_samples = set(n_id_dict.keys()) - set(p_id_dict.keys())
    bad_samples = set(p_id_dict.keys()) - set(n_id_dict.keys())
    vid = list(good_samples)[0]
    pred_label = labels[n_pred[n_id_dict[vid]]]
    tgt_lable = labels[n_tgt[n_id_dict[vid]]]
    # pred_label = labels[p_pred[p_id_dict[vid]]]
    # tgt_lable = labels[p_tgt[p_id_dict[vid]]]

    with TimerBlock("Good Luck") as block:
        args = parser_args.parser_args(block, config_path=n_config)
        edge_chose = args.data
        if args.data == 'knitics':
            is_3d = False
        else:
            is_3d = True
        edge = edge_dict[edge_chose]
        data_loader_train, data_loader_val = data_choose(args, block)
        sample_name = data_loader_val.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]

        index = sample_id.index(vid[:-9])
        data = data_loader_val.dataset[index][0]
        data = torch.tensor(data).unsqueeze(0)
        tag = 'DSTA: ' + pred_label +'\n' + 'Ours:' + tgt_lable +'\n' + 'TGT:' + tgt_lable
        vis_all(data=data.numpy(), edge=edge, is_3d=is_3d, tag=tag, pause=0.01)




    print('ok')











if __name__ == '__main__':
    Statistic_compare(npath_0='./model_0/wrong_path_pre_true.txt', npath_1='./model_1/wrong_path_pre_true.txt')
    print('end')