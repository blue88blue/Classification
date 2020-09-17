from dataset.dataset_classifier import myDataset
from utils import utils
from settings import basic_setting
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import csv
import random
import numpy as np
from PIL import Image
import time
import torchsummary
from torchvision.utils import save_image
# models
from model.classifer_base import classifer_base


def main(args, num_fold=0):
    # 模型选择
    model = classifer_base(args.network, args.pretrained, args.n_class, args.in_channel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.mode == "train" and num_fold <= 1:
        torchsummary.summary(model, (3, args.crop_size[0], args.crop_size[1]))  # #输出网络结构和参数量
    print(f'   [network: {args.network}  device: {device}]')

    if args.mode == "train":
        train(model, device, args, num_fold=num_fold)
    elif args.mode == "test":
        test(model, device, args, num_fold=num_fold)
    else:
        raise NotImplementedError





def train(model, device, args, num_fold=0):
    dataset_train = myDataset(args.data_root, args.target_root, args.crop_size,  "train",
                                 k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    num_train_data = len(dataset_train)  # 训练数据大小
    dataset_val = myDataset(args.val_data_root, args.val_target_root, args.crop_size, "val",
                               k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    num_train_val = len(dataset_val)  # 验证数据大小

    ####################

    writer = SummaryWriter(log_dir=args.log_dir[num_fold], comment=f'tb_log')
    # opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(torch.tensor(args.class_weight, device=device))

    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        lr = utils.poly_learning_rate(args, opt, epoch)  # 学习率调节

        with tqdm(total=num_train_data, desc=f'[Train] fold[{num_fold}/{args.k_fold}] Epoch[{epoch + 1}/{args.num_epochs} LR{lr:.8f}] ', unit='img') as pbar:
            for batch in dataloader_train:
                step += 1
                # 读取训练数据
                image = batch["image"]
                label = batch["label"]
                assert len(image.size()) == 4
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                # 前向传播
                opt.zero_grad()
                outputs = model(image)
                main_out = outputs["main_out"]

                # 计算损失
                totall_loss = criterion(main_out, label)
                if args.aux:  # 计算辅助损失函数
                    for aux_p in outputs["aux_out"]:
                        auxloss = criterion(aux_p, label)
                        totall_loss += auxloss * args.aux_weight

                totall_loss.backward()
                opt.step()

                if step % 5 == 0:
                    writer.add_scalar("Train/Totall_loss", totall_loss.item(), step)
                pbar.set_postfix(**{'loss': totall_loss.item()})  # 显示loss
                pbar.update(image.size()[0])

        if (epoch+1) % args.val_step == 0:
            # 保存模型
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{epoch + 1}.pth'))
            # 验证
            auc = val(model, dataloader_val, num_train_val, device, args)

            writer.add_scalar("Val/mean AUC", auc['mean'], step)
            for key in args.label_names:
                writer.add_scalar(f"Val/AUC({key})", auc[key], step)

            # 写入csv文件
            val_result = [num_fold, epoch+1, auc['mean']]
            with open(args.val_result_file, "a") as f:
                w = csv.writer(f)
                w.writerow(val_result)


def val(model, dataloader, num_train_val,  device, args):
    all_pred = []
    all_label = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=num_train_val, desc=f'VAL', unit='img') as pbar:
            for batch in dataloader:
                image = batch["image"]
                label = batch["label"]
                assert len(image.size()) == 4
                image = image.to(device, dtype=torch.float32)
                all_label.append(label)

                outputs = model(image)
                pred = torch.softmax(outputs["main_out"], dim=1)
                all_pred.append(pred)

                pbar.update(image.size()[0])
    all_pred = torch.cat(all_pred, 0).cpu()
    all_label = torch.cat(all_label, 0).cpu()

    auc = utils.multi_label_AUC(all_pred, all_label, args.label_names, args.plot_save_dir)
    print(f"mean AUC = {auc['mean']}")
    return auc



def test(model, device, args, num_fold=0):
    # 导入模型, 选取每一折的最优模型
    if os.path.exists(args.val_result_file):
        with open(args.val_result_file, "r") as f:
            reader = csv.reader(f)
            val_result = list(reader)
        best_epoch = utils.best_model_in_fold(val_result, num_fold)
    else:
        best_epoch = args.num_epochs
    # 导入模型
    model_dir = os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{best_epoch}.pth')
    model.load_state_dict(torch.load(model_dir, map_location=device))
    print(f'\rtest model loaded: [fold:{num_fold}] [best_epoch:{best_epoch}]')

    dataset_test = myDataset(args.val_data_root, args.val_target_root, args.crop_size, "test",
                                k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold)
    dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    all_pred = []
    all_label = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset_test), desc=f'TEST fold {num_fold}/{args.k_fold}', unit='img') as pbar:
            for batch in dataloader:
                image = batch["image"]
                label = batch["label"]
                assert len(image.size()) == 4
                image = image.to(device, dtype=torch.float32)
                all_label.append(label)

                outputs = model(image)
                pred = torch.softmax(outputs["main_out"], dim=1)
                all_pred.append(pred)

                pbar.update(image.size()[0])

    all_pred = torch.cat(all_pred, 0).cpu()
    all_label = torch.cat(all_label, 0).cpu()

    auc = utils.multi_label_AUC(all_pred, all_label, args.label_names, args.plot_save_dir, num_fold)
    utils.save_print_score(auc, args.test_result_file, args.label_names)



if __name__ == "__main__":

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    # cudnn.benchmark = True

    args = basic_setting()
    assert args.k_fold != 1
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

    # 交叉验证所需， 文件名列表
    if (not os.path.exists(args.dataset_file_list)) and (args.k_fold is not None):
        utils.get_dataset_filelist(args.data_root, args.dataset_file_list)

    mode = args.mode
    if args.k_fold is None:
        print("k_fold is None")
        if mode == "train_test":
            args.mode = "train"
            print("###################### Train Start ######################")
            main(args)
            args.mode = "test"
            print("###################### Test Start ######################")
            main(args)
        else:
            main(args)
    else:
        if mode == "train_test":
            print("###################### Train & Test Start ######################")

        if mode == "train" or mode == "train_test":
            args.mode = "train"
            print("###################### Train Start ######################")
            for i in range(args.start_fold, args.end_fold):
                torch.cuda.empty_cache()
                main(args, num_fold=i + 1)

        if mode == "test" or mode == "train_test":
            args.mode = "test"
            print("###################### Test Start ######################")
            for i in range(args.start_fold, args.end_fold):
                main(args, num_fold=i + 1)








