from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
import torch
import albumentations as A


class StomatchData(Dataset):
    """
    Data loader for binary-segmentation training
    """
    def __init__(self, metadata_file='dir.json', test_fold=0, mode='train', img_size=(320, 320)):
        # split dataset into 5 folds
        self.samples = []
        self.img_size = img_size
        self.mode = mode
        # self.segmentation_classes = segmentation_classes

        with open(metadata_file) as f:
            dirs = json.load(f)['dirs']
        
        for dir_info in dirs:
            type = dir_info['type']
            position_label = dir_info.get('position_label', -1)
            damage_label = dir_info.get('damage_label', -1)
            seg_label = dir_info.get('segmentation_label', 0)
            location = dir_info['location']
            img_folder_name = dir_info.get('img_folder_name', '')
            img_file_extension = dir_info.get('img_file_extension', '')
            mask_folder_name = dir_info.get('mask_folder_name', '')
            mask_file_extension = dir_info.get('mask_file_extension', '')

            files_name = os.listdir(location + '/' + img_folder_name)
            files_name = sorted(files_name)

            img_per_fold = int(len(files_name) / 5)
            if self.mode == 'train':
                names = files_name[:img_per_fold * test_fold] + files_name[img_per_fold * (test_fold+1):]
            else:
                names = files_name[img_per_fold * test_fold : img_per_fold * (test_fold+1)]

            for fn in names:
                img_path = location + '/' + img_folder_name + '/' + fn
                mask_path = location + '/' + mask_folder_name + '/' + fn.replace(img_file_extension, mask_file_extension)
                self.samples.append([img_path, mask_path, position_label, damage_label, seg_label])


    def aug(self, image, mask):
        img_size = self.img_size
        if self.mode == 'train':
            t1 = A.Compose([A.Resize(img_size[0], img_size[1]),])
            resized = t1(image=image, mask=mask)
            image = resized['image']
            mask = resized['mask']
            t = A.Compose([                
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.7),
                A.Rotate(interpolation=cv2.BORDER_CONSTANT, p=0.7),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0.5, scale_limit=0.2, p=0.7),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0, scale_limit=(-0.1, 0.1), rotate_limit=0, p=0.35),
                A.MotionBlur(p=0.2),
                A.HueSaturationValue(p=0.2),                
            ], p=0.5)

        elif self.mode == 'test':
            t = A.Compose([
                A.Resize(img_size[0], img_size[1])
            ])

        return t(image=image, mask=mask)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, position_label, damage_label, seg_label = self.samples[idx]
        # print(img_path, mask_path)

        img = cv2.imread(img_path).astype(np.float32)

        if mask_path is not None:
            mask = cv2.imread(mask_path).astype(np.float32)
        else:
            mask = img

        augmented = self.aug(img, mask)
        img = augmented['image']
        mask = augmented['mask']

        img = torch.from_numpy(img.copy())
        img = img.permute(2, 0, 1)
        mask = torch.from_numpy(mask.copy())
        mask = mask.permute(2, 0, 1)

        img /= 255.
        mask = mask.mean(dim=0, keepdim=True)/255.

        mask[mask <= 0.5] = 0
        mask[mask > 0.5] = 1

        return img, mask

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

    dataset = StomatchData(metadata_file='fake_dir.json')
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

if __name__=='__main__':
    d = Data(metadata_file='fake_dir.json')