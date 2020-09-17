from dataset.dataset_classifier import myDataset
from torchvision.utils import save_image
import torch
from settings import basic_setting
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_root = '/home/sjh/dataset/LAG/image'
    target_root = "/home/sjh/dataset/LAG/1200_Fovea_locations.csv"
    crop_size = (256, 256)   # (h, w)
    mode = "train"
    dataset = myDataset(data_root, target_root, crop_size, mode)

    batch = dataset[15]
    image = batch["image"]
    label = batch["label"]

    save_image(image, "image.jpg")
    # save_image(label, "label.png", normalize=True)

    print(image.size())
    print(label)




