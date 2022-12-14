import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler, RandomSampler
from utilizes.augment import NoAugmenter, Augmenter
from torch import distributed as dist
import albumentations as A
import warnings
import cv2
import torch
warnings.filterwarnings('ignore')


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_paths, gt_paths, img_size, transforms=None, mode='train'):
        self.img_size = img_size
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.images = sorted(self.image_paths)
        self.gts = sorted(self.gt_paths)
        self.filter_files()
        self.size = len(self.images)
        self.transforms = transforms
        self.mode = mode
        self.img_size = img_size
        

    # def __getitem__(self, index):
    #     image = self.rgb_loader(self.images[index])
    #     gt = self.binary_loader(self.gts[index])
        
    #     if self.transforms is not None:
    #         couple_transform = self.transforms(image=image, mask=gt)
    #         image = couple_transform['image']
    #         gt = couple_transform['mask']
            
    #     elif transforms is None or self.mode == 'test' or self.mode ==' valid':

    #         no_augment = NoAugmenter(self.img_size)
    #         couple_transform = no_augment(image=image, mask=gt)
    #         image = couple_transform['image']
    #         gt = couple_transform['mask']
            
    #     return image, gt
    
    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        gt_paths = self.gt_paths[idx]
        image_ = np.array(Image.open(image_paths).convert("RGB"))
        mask = np.array(Image.open(gt_paths).convert("L"))
        
        augmented = self.transforms(image=image_, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        mask_resize = mask
        mask = mask / 255

        if self.mode == "train":
            mask = cv2.resize(mask, (self.img_size, self.img_size))
        elif self.mode == "val":
            mask_resize = cv2.resize(mask, (self.img_size, self.img_size),interpolation = cv2.INTER_NEAREST)
            mask_resize = mask_resize[:, :, np.newaxis]

            mask_resize = mask_resize.astype("float32")
            mask_resize = mask_resize.transpose((2, 0, 1))

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]

        mask = mask.astype("float32")
        mask = mask.transpose((2, 0, 1))

        if self.mode == "train":
            return np.asarray(image), np.asarray(mask)

        elif self.mode == "test":
            return (
                np.asarray(image),
                np.asarray(mask),
                os.path.basename(image_paths),
                np.asarray(image_),
            )
        else:
            return (
                np.asarray(image),
                np.asarray(mask),
                np.asarray(mask_resize),
            )


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
            img = np.array(img)
            return img

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("L")
            img = np.array(img)
            return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.img_size or w < self.img_size:
            h = max(h, self.img_size)
            w = max(w, self.img_size)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class NeoDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_paths, gt_paths, img_size, transforms=None, mode='train'):
        self.img_size = img_size
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.images = sorted(self.image_paths)
        self.gts = sorted(self.gt_paths)
        self.filter_files()
        self.size = len(self.images)
        self.transforms = transforms
        self.mode = mode
        self.img_size = img_size
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        gt_path = self.gts[idx]
        image_ = np.array(cv2.imread(image_path)[:,:,::-1])
        mask = np.array(cv2.imread(gt_path))
        neo_gt = np.all(mask == [0, 0, 255], axis=-1).astype('float')
        non_gt = np.all(mask == [0, 255, 0], axis=-1).astype('float')

        augmented = self.transforms(image=image_, neo=neo_gt, non=non_gt)
        
        image, neo, non = augmented["image"], augmented["neo"], augmented['non']

        mask = np.stack([neo, non], axis=-1)
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate([mask, background], axis=-1)
        mask_resize = mask
        
        # if self.mode == "train":
        #     mask = cv2.resize(mask, (self.img_size, self.img_size))
        # elif self.mode == "val":
        #     mask_resize = cv2.resize(mask, (self.img_size, self.img_size),interpolation = cv2.INTER_NEAREST)

        #     mask_resize = mask_resize.astype("float32")
        #     mask_resize = mask_resize.transpose((2, 0, 1))

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask.astype("float32")
        mask = mask.transpose((2, 0, 1))

        if self.mode == "train":
            return np.asarray(image), np.asarray(mask)

        elif self.mode == "test":
            return (
                np.asarray(image),
                np.asarray(mask),
                os.path.basename(image_path),
                np.asarray(image_),
            )
        else:
            return (
                np.asarray(image),
                np.asarray(mask),
                np.asarray(mask_resize),
            )

    def read_mask(self, mask_path):

        image = cv2.cvtColor(mask_path, cv2.COLOR_BGR2HSV)
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)

        red_mask = lower_mask + upper_mask
        red_mask[red_mask != 0] = 2
        
        # boundary RED color range values; Hue (36 - 70)
        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255,255))
        green_mask[green_mask != 0] = 1
        
        full_mask = cv2.bitwise_or(red_mask, green_mask)
        full_mask = full_mask.astype(np.uint8)
        full_mask= cv2.dilate(full_mask, np.ones((5,5)), iterations=1)
        full_mask = cv2.erode(full_mask, np.ones((5,5)), iterations=1)  
        return full_mask.astype(np.uint8)

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
            img = np.array(img)
            return img

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("L")
            img = np.array(img)
            return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.img_size or w < self.img_size:
            h = max(h, self.img_size)
            w = max(w, self.img_size)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_neo_loader(
    image_paths,
    gt_paths,
    transforms,
    batchsize,
    img_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    mode='train',
    use_ddp=False
):

    dataset = NeoDataset(image_paths, gt_paths, img_size, transforms=transforms, mode=mode)
    if use_ddp:
        sampler = DistributedSampler(dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler
        )
    else:
        data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return data_loader


def get_loader(
    image_paths,
    gt_paths,
    transforms,
    batchsize,
    img_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    mode='train',
    use_ddp=False
):

    dataset = PolypDataset(image_paths, gt_paths, img_size, transforms=transforms, mode=mode)
    if use_ddp:
        sampler = DistributedSampler(dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler
        )
    else:
        data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return data_loader

if __name__ == '__main__':
    image_root = '/home/admin_mcn/hungpv/polyps/dataset/KCECE/TrainDataset/images'
    gt_root = '/home/admin_mcn/hungpv/polyps/dataset/KCECE/TrainDataset/masks'
    
    image_paths = [os.path.join(image_root, i) for i in os.listdir(image_root)]
    gt_paths = [os.path.join(gt_root, i) for i in os.listdir(gt_root)]
    augment = Augmenter(prob=1)
    dataset = PolypDataset(image_paths, gt_paths, img_size=352, transforms=augment, mode='val')
    img, gt = dataset.__getitem__(0)
    dataloader = get_loader(image_paths, gt_paths, transforms=augment, batchsize=2, img_size=352)
    for i, (imgs, gts) in enumerate(dataloader):
        
        print(imgs.shape)
        print(gts.shape)
        if i == 3:
            break
    

