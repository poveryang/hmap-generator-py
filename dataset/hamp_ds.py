import random
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader

from dataset.hmap_trans import generate_hmap

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HeatMapDataset(Dataset):
    def __init__(self, root_dir, mode, input_size=(512, 512)):
        self.eval = (mode != 'train')
        self.root_dir = Path(root_dir)
        self.input_size = input_size
        self.img_paths, self.labels = self.load_items(mode)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image1 = self.random_jitter(image)

        instances = self.labels[idx]
        hmap = generate_hmap(instances, image1)

        image_tensor, hmap_tensor = self.geometric_trans(image1, hmap)
        return image_tensor, hmap_tensor

    def __len__(self):
        return len(self.img_paths)

    def load_items(self, mode):
        img_paths, labels = [], []
        label_file = self.root_dir / mode / f'{mode}.txt'
        f = open(label_file, mode='r', encoding='utf-8')
        for item in f.readlines():
            # extract image path
            item_parts = item.strip().split(';')
            img_path = self.root_dir / mode / item_parts[0]
            # extract instances
            instances_str = item_parts[1:]
            instances = []
            for instance in instances_str:
                instance_parts = instance.split(",")
                rrect = [float(r) for r in instance_parts[:-1]]
                label_id = int(instance_parts[-1])
                instance = rrect + [label_id]
                instances.append(instance)
            # load to list
            img_paths.append(img_path)
            labels.append(instances)
        return img_paths, labels

    def random_jitter(self, image):
        # To tensor
        image = TF.to_tensor(image)

        # Random adjust image's brightness, contrast, blur
        if random.random() < 0.2 and not self.eval:
            factor = random.uniform(0.5, 2)
            image = TF.adjust_brightness(image, factor)

        if random.random() < 0.2 and not self.eval:
            factor = random.uniform(0.5, 2)
            image = TF.adjust_contrast(image, factor)

        if random.random() < 0.2 and not self.eval:
            ksize = random.choice((3, 7, 11))
            image = TF.gaussian_blur(image, kernel_size=[ksize, ksize])

        # To PIL Image
        image = TF.to_pil_image(image)
        return image

    def geometric_trans(self, image, hmap):
        # ToTensor
        image = TF.to_tensor(image)
        hmap = TF.to_tensor(hmap)
        # Normalize
        image = TF.normalize(image, mean=[0.4330], std=[0.2349])

        # Random horizontal flip
        if random.random() < 0.5 and not self.eval:
            image = TF.hflip(image)
            hmap = TF.hflip(hmap)

        # Random vertical flip
        if random.random() < 0.5 and not self.eval:
            image = TF.vflip(image)
            hmap = TF.vflip(hmap)

        # Random rotation
        if random.random() < 0.2 and not self.eval:
            angle = random.randint(-90, 90)
            image = TF.rotate(image, angle)
            hmap = TF.rotate(hmap, angle)

        # Random crop resize
        if random.random() < 0.8 and not self.eval:
            image, hmap = self.crop_resize(image, hmap)
        else:
            image, hmap = self.aspect_resize(image, hmap)
        return image, hmap

    def crop_resize(self, image, hmap, crop_prob=0.2):
        # random crop
        if random.random() < crop_prob:
            img_size = list(image.shape[-2:])

            crop_rate = random.uniform(0.7, 0.9)
            crop_size = [int(img_size[0] * crop_rate), int(img_size[1] * crop_rate)]

            offset_rate = random.uniform(-0.2, 0.2)
            dy, dx = [int(img_size[0] * offset_rate), int(img_size[1] * offset_rate)]

            image = TF.crop(image, dy, dx, crop_size[0], crop_size[1])
            hmap = TF.crop(hmap, dy, dx, crop_size[0], crop_size[1])
        # resize
        image = TF.resize(image, [self.input_size[0], self.input_size[1]])
        hmap = TF.resize(hmap, [self.input_size[0], self.input_size[1]])
        return image, hmap

    def aspect_resize(self, image, hmap):
        dst_image = torch.full((1, self.input_size[0], self.input_size[1]), 0.5, dtype=torch.float32)
        dst_hmap = torch.zeros((3, self.input_size[0], self.input_size[1]), dtype=torch.float32)

        img_h, img_w = image.shape[-2:]
        scale = min(self.input_size[0] / img_h, self.input_size[1] / img_w)
        scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)
        dx = (self.input_size[1] - scaled_w) // 2
        dy = (self.input_size[0] - scaled_h) // 2
        scaled_image = TF.resize(image, [scaled_h, scaled_w])
        scaled_hmap = TF.resize(hmap, [scaled_h, scaled_w])

        dst_image[:, dy:dy + scaled_h, dx:dx + scaled_w] = scaled_image
        dst_hmap[:, dy:dy + scaled_h, dx:dx + scaled_w] = scaled_hmap

        return dst_image, dst_hmap


def get_dataloader(conf):
    train_dataset = HeatMapDataset(
        conf.root_dir,
        mode='train',
        input_size=conf.input_size)
    val_dataset = HeatMapDataset(
        conf.root_dir,
        mode='test',
        input_size=conf.input_size)
    sample_dataset = HeatMapDataset(
        conf.root_dir,
        mode='sample',
        input_size=conf.input_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=conf.batch_size,
                              shuffle=True,
                              num_workers=conf.num_workers,
                              pin_memory=True,
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=conf.batch_size,
                            shuffle=False,
                            num_workers=conf.num_workers,
                            pin_memory=True,
                            )
    sample_loader = DataLoader(sample_dataset,
                               batch_size=4,
                               shuffle=False,
                               pin_memory=True,
                               )
    return train_loader, val_loader, sample_loader


def sample_loader(conf):
    sample_dataset = HeatMapDataset(conf.root_dir, mode='sample', input_size=(400, 640))
    sample_dataloader = DataLoader(sample_dataset, batch_size=4, shuffle=False, pin_memory=True)
    return sample_dataloader


def blend_image_hmap_tensor(img, hmap, alpha=0.5):
    from torchvision.utils import make_grid
    img = img * 0.2349 + 0.4330
    hmap = torch.sigmoid(hmap)
    blended_batch = img * alpha + hmap * (1 - alpha)
    blended_batch = (blended_batch - blended_batch.min()) / (blended_batch.max() - blended_batch.min())
    blended_grid = make_grid(blended_batch, nrow=4)
    return blended_grid


if __name__ == '__main__':
    from tqdm import tqdm

    dataset = HeatMapDataset(
        '/home/pover/Datasets/barcode',
        mode='train',
        input_size=(400, 640))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=8)

    for i, (im, hm) in enumerate(tqdm(dataloader)):
        # blended = blend_image_hmap_tensor(img, hmap)
        # plt.figure(figsize=(6.4 * 4, 4))
        # plt.imshow(blended.permute(1, 2, 0))
        # plt.show()
        pass
