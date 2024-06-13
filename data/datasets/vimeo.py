import os
import numpy as np
import h5py
from PIL import Image
from torch.utils.data import Dataset
import torch

class VIMEO(Dataset):
    def __init__(self, path, train=True, transform=None, add_noise=False):
        assert os.path.exists(
            path), 'Invalid path to VIMEO data set: ' + path
        self.path = path
        self.transform = transform
        self.add_noise = add_noise

        if train:
            filename = 'train.h5'
        else:
            filename = 'test.h5'

        self.h5_file = h5py.File(os.path.join(path, filename), 'r')
        self.index_dset = self.h5_file['index'][:]
        self.img_dset = self.h5_file['images']

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        folder_path = self.index_dset[ind].decode('utf-8').rsplit('/', 1)[0]
        inds = [i for i, path in enumerate(self.index_dset) if path.decode('utf-8').startswith(folder_path)]
        imgs = [Image.fromarray(self.img_dset[i]) for i in inds]

        if self.transform is not None:
            imgs = self.transform(imgs)

        if self.add_noise:
            imgs = imgs + (torch.rand_like(imgs)-0.5) / 256.

        return imgs

    def __len__(self):
        # total number of videos
        unique_folders = set([path.decode('utf-8').rsplit('/', 1)[0] for path in self.index_dset])
        return len(unique_folders)
