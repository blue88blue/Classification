from dataset.dataset_classifier import PredictDataset
from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings import basic_setting
import numpy as np
from tqdm import tqdm
import csv
#models
from model.classifer_base import classifer_base


# #################################### predict settings 预测提交结果 ####################################
pred_data_root = '/home/sjh/dataset/REFUGE2_ROI_768/Refuge2_Validation'
pred_mask_root = "/home/sjh/dataset/REFUGE2_ROI_768/ROI_val_pred_mask"
pred_dir = 'classification_results.csv'
model_CPepoch = 10
# #################################### predict settings 预测提交结果 ####################################


def pred(model, device, args, num_fold=0):
    dataset_pred = PredictDataset(pred_data_root, args.crop_size)
    num_data = len(dataset_pred)
    dataloader_pred = DataLoader(dataset_pred, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,pin_memory=True, drop_last=False)
    with open(pred_dir[num_fold], "w") as f:
        w = csv.writer(f)
        w.writerow(['FileName']+args.label_names)

    model.eval()
    with tqdm(total=num_data, desc=f'preddict', unit='img') as pbar:
        for batch in dataloader_pred:
            image = batch["image"]
            file_name = batch["file_name"]
            image = image.to(device, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(image)
                pred = torch.softmax(outputs["main_out"], dim=1)

            # 保存预测结果
            for i in range(image.shape[0]):
                with open(pred_dir[num_fold], "a") as f:
                    w = csv.writer(f)
                    w.writerow([file_name[i]+".jpg"]+[float(pred[i, k]) for k in range(pred.size()[-1])])
            pbar.update(image.shape[0])




if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = basic_setting()

    # 模型选择
    model = classifer_base(backbone=args.network, pretrained_base=args.pretrained, n_class=args.n_class, in_channel=args.in_channel)

    pred_dir = [os.path.join(args.dir, pred_dir)]  # 预测文件

    if args.k_fold == None:
        model.to(device)
        model_dir = os.path.join(args.checkpoint_dir[0], f'CP_epoch{model_CPepoch}.pth')  # 最后一个epoch模型
        model.load_state_dict(torch.load(model_dir, map_location=device))
        print("model loaded!")

        pred(model, device, args)
    else:
        for i in range(args.start_fold, args.end_fold):
            pred_dir.append(os.path.join(args.dir, f'classification_results_{i+1}.csv'))
            model.to(device)
            model_dir = os.path.join(args.checkpoint_dir[i+1], f'CP_epoch{model_CPepoch}.pth')  # 最后一个epoch模型
            model.load_state_dict(torch.load(model_dir, map_location=device))
            print("model loaded!")

            pred(model, device, args, i+1)

