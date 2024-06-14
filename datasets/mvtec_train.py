import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from self_sup_tasks import patch_ex

URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle','cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
           'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']


class SelfSupMVTecDataset(Dataset):
    def __init__(self, root_path='../data', class_name='bottle', is_train=True,
                 low_res=256, transform=None,
                 self_sup_args={}):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.root_path = root_path
        self.class_name = class_name
        self.is_train = is_train
        self.mvtec_folder_path = os.path.join(root_path)

        self.transform = transform
        self.norm_transform = T.Compose([
            T.ToTensor(),
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()])
        self.mask1_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor()])

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder(low_res)

        self.self_sup = is_train
        self.self_sup_args = self_sup_args
        self.prev_idx = np.random.randint(len(self.x))

    def __getitem__(self, idx):
        x, y, mask1 = self.x[idx], self.y[idx], self.mask[idx]
        image = x
        image = self.norm_transform(image)
        x = np.asarray(x)
        p = self.x[self.prev_idx]
        if self.transform is not None:
            p = self.transform(p)
        p = np.asarray(p)
        x, mask, src_object_mask = patch_ex(x, p, **self.self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float().squeeze(dim=0)
        mask = mask.numpy()
        mask = Image.fromarray(mask)
        mask0 = self.mask_transform(mask)
        mask_1 = self.mask1_transform(mask)
        self.prev_idx = idx
        mask1 = self.transform(mask1)
        x = self.norm_transform(x)
        return image, x, y, mask1, mask0, mask_1

    def __len__(self):
        return len(self.x)

    def configure_self_sup(self, on=True, self_sup_args={}):
        self.self_sup = on
        self.self_sup_args.update(self_sup_args)

    def load_dataset_folder(self, low_res):
        phase = 'train' if self.is_train else 'test'
        x_paths, y, mask_paths = [], [], []

        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, phase)
        gt_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            ################ random chose 2 images
            indices = torch.randint(0, len(img_fpath_list), (2,))
            for i in range(len(indices)):
                x_paths.append(img_fpath_list[indices[i]])

            if img_type == 'good':
                y.extend([0] * len(x_paths))
                mask_paths.extend([None] * len(x_paths))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)

        assert len(x_paths) == len(y), 'number of x and y should be same'

        transform = T.Resize(low_res, Image.ANTIALIAS)
        xs = []
        for path in x_paths:
            xs.append(transform(Image.open(path).convert('RGB')))


        mask_transform = T.Compose([T.Resize(low_res, Image.NEAREST), T.ToTensor()])
        masks = []
        for path in mask_paths:
            if path is None:
                masks.append(torch.zeros((1, low_res, low_res)))
            else:
                masks.append(mask_transform(Image.open(path)))
        return list(xs), list(y), torch.stack(masks)


    # copied from https://github.com/byungjae89/SPADE-pytorch/blob/master/src/datasets/mvtec.py
    def download(self, url=URL):
        """Download dataset if not exist"""

        if not os.path.exists(self.mvtec_folder_path):
            tar_file_path = self.mvtec_folder_path + '.tar.xz'
            if not os.path.exists(tar_file_path):
                download_url(url, tar_file_path)
            print('unzip downloaded dataset: %s' % tar_file_path)
            tar = tarfile.open(tar_file_path, 'r:xz')
            tar.extractall(self.mvtec_folder_path)
            tar.close()

        return


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)