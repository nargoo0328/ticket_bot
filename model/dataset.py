import os
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch
import math
import json

def parse_data(data):
    labels = []
    for s in data:
        temp = [0]*6
        for i in range(len(s)):
            temp [i] = ord(s[i])-96
        labels.append(temp)
    return np.array(labels)

class captcha_dataset(Dataset):
    def __init__(self,img_path,labels_path, mode='train' ,cuda=False):
        self.cuda = cuda
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.8982),(0.1465)),
        ])
        self.img_path = img_path
        self.imgs = sorted(os.listdir(self.img_path),key=len)
        if '.npy' in labels_path:
            data = np.load(labels_path)
        else:
            f = open(labels_path)
            data = json.load(f)
            f.close()
        self.labels = parse_data(data)
        
        assert mode == 'train' or mode == 'val' or mode == 'tixcraft'
        slice = math.floor(len(self.imgs)*0.85)
        if mode == 'train':
            self.imgs = self.imgs[:slice]
            self.labels = self.labels[:slice]
        elif mode == 'val':
            self.imgs = self.imgs[slice:]
            self.labels = self.labels[slice:]

    def __getitem__(self, index):
        file_name = self.imgs[index]
        # img = Image.open(os.path.join(self.img_path,file_name))
        # # applying grayscale method
        # img = ImageOps.grayscale(img)
        img = cv2.imread(os.path.join(self.img_path,file_name),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(128,96),interpolation=cv2.INTER_NEAREST)
        img = self.transform(img)
        label = self.labels[index]
        if self.cuda:
            img = img.cuda()
            label = torch.from_numpy(label).cuda().to(torch.long)
        sample = {'image': img, 'label': label, 'file_name':os.path.join(self.img_path,file_name)}
        return sample

    def __len__(self):
        return len(self.labels)