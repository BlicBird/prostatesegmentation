import os
import torch
import numpy as np
import h5py
import random
from torchvision import transforms as tr
from torchvision.transforms import Compose


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, random=False, seed=18):
        self.h5_file = h5py.File(file_path, 'r')
        self.cases = sorted(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.cases_num = len(self.cases)
        self.frames_per_case = [len([k for k in self.h5_file.keys() if k.split('_')[1] == idx]) / 4 for idx in
                                self.cases]
        self.random = random
        self.seed = seed

    def __len__(self):
        return self.cases_num

    def __getitem__(self, index):
        if self.random:
            random.seed(self.seed)
        idx_frame = random.randint(0, self.frames_per_case[index] - 1)
        idx_label = random.randint(0, 2)
        frame = torch.unsqueeze(
            torch.tensor(self.h5_file['/frame_%04d_%03d' % (index, idx_frame)][()].astype('float32')), dim=0)
        label = torch.unsqueeze(
            torch.tensor(self.h5_file['/label_%04d_%03d_%02d' % (index, idx_frame, idx_label)][()].astype('float32')),
            dim=0)
        return (frame, label)


class H5Dataset2(torch.utils.data.Dataset):
    def __init__(self, file_path, random=False, seed=18):
        self.h5_file = h5py.File(file_path, 'r')
        self.cases = sorted(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.cases_num = len(self.cases)
        self.frames_per_case = [len([k for k in self.h5_file.keys() if k.split('_')[1] == idx]) / 4 for idx in
                                self.cases]
        self.random = random
        self.seed = seed

    def __len__(self):
        return self.cases_num

    def __getitem__(self, index):
        if self.random:
            random.seed(self.seed)
        idx_frame = random.randint(0, self.frames_per_case[index] - 1)
        frame = torch.unsqueeze(
            torch.tensor(self.h5_file['/frame_%04d_%03d' % (index, idx_frame)][()].astype('float32')), dim=0)
        labels = [torch.unsqueeze(
            torch.tensor(self.h5_file['/label_%04d_%03d_%02d' % (index, idx_frame, idx_label)][()].astype('float32')),
            dim=0) for idx_label in range(3)]
        label = (torch.stack(labels, dim=0).sum(dim=0) >= 1.5).long()
        return (frame, label.float())


class H5Dataset_classification(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.h5_file = h5py.File(file_path, 'r')
        self.cases = sorted(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.cases_num = len(self.cases)
        self.frames_per_case = [len([k for k in self.h5_file.keys() if k.split('_')[1] == idx]) / 4 for idx in
                                self.cases]

    def __len__(self):
        return self.cases_num

    def __getitem__(self, index):
        idx_frame = random.randint(0, self.frames_per_case[index] - 1)
        frame = torch.unsqueeze(
            torch.tensor(self.h5_file['/frame_%04d_%03d' % (index, idx_frame)][()].astype('float32')), dim=0)
        labels = [torch.unsqueeze(
            torch.tensor(self.h5_file['/label_%04d_%03d_%02d' % (index, idx_frame, idx_label)][()].astype('float32')),
            dim=0) for idx_label in range(3)]
        segmentation = (torch.stack(labels, dim=0).sum(dim=0) >= 1.5).long()
        count = 0
        for i in labels:
            if torch.count_nonzero(i) != 0:
                count += 1
        if count >= 2:
            label = torch.tensor([1, 0]).float()
        else:
            label = torch.tensor([0, 1]).float()
        return (frame, label, segmentation)


class Transform(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.transform = transform
        self.data = data
        self.data_num = len(self.data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        frame = self.transform(self.data[index][0])
        label = self.transform(self.data[index][1])
        return (frame, label)


class Transform_classification(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.transform = transform
        self.data = data
        self.data_num = len(self.data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        frame = self.transform(self.data[index][0])
        label = self.data[index][1]
        return (frame, label)


class Classification_screen(torch.utils.data.Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels
        self.num_frames = len(frames)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        frame = self.frames[index]
        label = self.labels[index]

        return frame, label

    




    
