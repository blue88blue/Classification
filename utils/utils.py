
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import csv
import os
import random
import h5py
import matplotlib.pyplot as plt

def AUC(pred, label, roc_image=None):
    assert len(pred.shape) == 1 and len(label.shape) == 1
    num_data = pred.shape[0]
    thresholds = [i / 100.0 for i in reversed(range(0, 101))]  # 阈值
    tpr = []
    fpr = []
    for th in thresholds:
        p = pred > th   # true or flase
        true_positive = torch.nonzero((p == label)*label).size()[0]
        flase_positive = torch.nonzero(p).size()[0] - true_positive

        all_positive = torch.nonzero(label).size()[0]  # 有病样本的数量
        all_negative = num_data - all_positive  # 正常样本的数量

        tpr_ = true_positive / all_positive  # 真正类率， sensitivity, recall
        fpr_ = flase_positive / all_negative  # 假正类率
        tpr.append(tpr_)
        fpr.append(fpr_)
    # 计算AUC
    auc = 0
    for i in range(len(tpr)-1):
        auc += (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2
    # 保存ROC曲线图
    if roc_image is not None:
        plt.cla()
        plt.title(f"AUC={auc:.5f}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.plot(np.array(fpr), np.array(tpr))
        plt.savefig(roc_image)

    return auc



def multi_label_AUC(pred, label, class_names, img_save_dir=None, num_fold=None):
    assert len(pred.size()) == 2 and len(label.size()) == 1
    auc = dict()
    m_auc = []
    for i in range(pred.shape[1]):
        pred_i = pred[:, i]
        label_i = (label == i).float()
        save_dir = None if img_save_dir is None else os.path.join(img_save_dir, class_names[i]+str(num_fold)+".png")
        auc_i = AUC(pred_i, label_i, save_dir)
        auc.update({class_names[i]: auc_i})
        m_auc.append(auc_i)
    m_auc = np.array(m_auc).mean()
    auc.update({"mean": m_auc})
    return auc

# 保存打印指标
def save_print_score(auc, file, label_names):
    result_AUC = [auc['mean']] + [auc[key] for key in label_names]
    title = ['mAUC'] + [name + "_AUC" for name in label_names  ]
    with open(file, "a") as f:
        w = csv.writer(f)
        w.writerow(title)
        w.writerow(result_AUC)

    print("\n##############Test Result##############")
    print(f"mean AUC: {auc['mean']}")
    for i in range(len(label_names)):
        print(f"{label_names[i]}: {auc[label_names[i]]}")



# 从验证指标中选择最优的epoch
def best_model_in_fold(val_result, num_fold):
    best_epoch = 0
    best_dice = 0
    for row in val_result:
        if str(num_fold) in row:
            if best_dice < float(row[2]):
                best_dice = float(row[2])
                best_epoch = int(row[1])
    return best_epoch



# 读取数据集目录内文件名，保存至csv文件
def get_dataset_filelist(data_root, save_file):
    file_list = os.listdir(data_root)
    random.shuffle(file_list)
    with open(save_file, 'w') as f:
        w = csv.writer(f)
        w.writerow(file_list)



def poly_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    lr = args.lr * (1 - epoch / args.num_epochs) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# one hot转成0,1,2,..这样的标签
def make_class_label(mask):
    b = mask.size()[0]
    mask = mask.view(b, -1)
    class_label = torch.max(mask, dim=-1)[0]
    return class_label

# 把0,1,2...这样的类别标签转化为one_hot
def make_one_hot(targets, num_classes):
    assert len(targets.size()) in [2, 3]
    targets = targets.unsqueeze(1)
    label = []
    for i in range(num_classes):
        label.append((targets == i).float())
    label = torch.cat(label, dim=1)
    return label



def save_h5(train_data, train_label, val_data, filename):
    file = h5py.File(filename, 'w')
    # 写入
    file.create_dataset('train_data', data=train_data)
    file.create_dataset('train_label', data=train_label)
    file.create_dataset('val_data', data=val_data)
    file.close()


def load_h5(path):
    file = h5py.File(path, 'r')
    train_data = torch.tensor(np.array(file['train_data'][:]))
    train_label = torch.tensor(np.array(file['train_label'][:]))
    val_data = torch.tensor(np.array(file['val_data'][:]))
    file.close()
    return train_data, train_label, val_data




