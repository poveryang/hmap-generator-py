import cv2
import glob
import numpy as np
import torch
from thop import profile
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import seaborn as sns

from dataset.hamp_ds import get_dataloader
from model.unet_pl import LitUNet
from configs.conf_linux import conf
import matplotlib.pyplot as plt
from utils.misc import blend_image_hmap_tensor
import torch.nn as nn


class HMapLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.2):
        super(HMapLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        diff = torch.abs(target - torch.sigmoid(pred))
        pos_mask = target >= 0.004
        neg_mask = target < 0.004
        pos_ratio = len(pos_mask) / (len(pos_mask) + len(neg_mask))
        alpha = max(self.alpha, pos_ratio)

        pos_loss = -torch.pow(diff[pos_mask], self.gamma) * torch.log(1-diff[pos_mask]) * target[pos_mask]
        neg_loss = -torch.pow(diff[neg_mask], self.gamma) * torch.log(1-diff[neg_mask])
        loss = alpha * torch.mean(pos_loss) + (1-alpha) * torch.mean(neg_loss)
        return loss


hmap_loss = HMapLoss()

model_path="./hmap_epoch=049_val_loss=0.000378.ckpt"
model = LitUNet(conf.model).load_from_checkpoint(model_path, model_conf=conf.model)
model.eval()


def batch_infer(model_path, batch_size=1):
    conf.data.batch_size = batch_size
    train_loader, val_loader, sample_loader = get_dataloader(conf.data)
    model = LitUNet(conf.model).load_from_checkpoint(model_path, model_conf=conf.model)
    model.eval()

    for i, (in_tensor, hmap_gt) in enumerate(val_loader):
        hmap_tensor = model(in_tensor)
        pos_loss, neg_loss = hmap_loss(hmap_tensor, hmap_gt)
        blend_image = blend_image_hmap_tensor(in_tensor, hmap_tensor, alpha=0.5)
        blend_image = np.array(TF.to_pil_image(blend_image))
        pass


# Inference on a single image
def infer_single_image(image_path):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))



    # preprocess
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_tensor = preprocess(image)

    # inference
    hmap_tensor = model(image_tensor)

    # postprocess
    hmap = postprocess(hmap_tensor)
    oned_hmap = hmap[:, :, 0]
    sns.heatmap(oned_hmap, ax=ax[1], cmap='jet', xticklabels=False, yticklabels=False)
    ax[1].set_title('heat map')
    save_path = image_path.replace('.png', '_hmap.png')

    # blend image and hmap
    blend_image = blend_image_hmap_tensor(image_tensor, hmap_tensor, alpha=0.5)
    blend_image = np.array(TF.to_pil_image(blend_image))

    ax[0].imshow(blend_image, cmap='gray')
    ax[0].set_title('image with hmap')
    ax[0].axis('off')

    # plt.show()
    fig.savefig(save_path, dpi=300)



def preprocess(image):
    image = (image / 255.0).astype(np.float32)
    image = (image - 0.4330) / 0.2349
    image = cv2.resize(image, (640, 400))
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    return image_tensor


def postprocess(hmap):
    hmap = torch.sigmoid(hmap)
    hmap = torch.permute(hmap, (0, 2, 3, 1))
    hmap = hmap[0].detach().numpy()
    # hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap = cv2.resize(hmap, (1280, 800))
    return hmap


def to_onnx(model_path, onnx_path):
    model = LitUNet(conf.model).load_from_checkpoint(model_path, model_conf=conf.model)
    model.eval()

    dummy_input = torch.randn(1, 1, 400, 640)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=11)


if __name__ == '__main__':
    # batch_infer("./test/ckpt/hmap_epoch=082_val_loss=0.000288.ckpt", batch_size=1)
    from multiprocessing import Pool

    files = glob.glob("/Users/yjunj/Downloads/一维码-有码/*.png")

    with Pool(16) as p:
        p.map(infer_single_image, files)


    # for file in files:
    #     infer_single_image(
    #         model_path="./hmap_epoch=049_val_loss=0.000378.ckpt",
    #         image_path=file
    #     )
    # infer_single_image(
    #     model_path="./hmap_epoch=049_val_loss=0.000378.ckpt",
    #     image_path="/Users/yjunj/Downloads/一维码-有码/20210324124313616.png"
    # )

    # to_onnx("ckpt/hmap-v3-e99-fp32.ckpt",
    #         "ckpt/hmap-v3-e99-fp32.onnx")
