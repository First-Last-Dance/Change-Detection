import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ChangeDetectionDataset(Dataset):
    def __init__(self, images_A, images_B, labels, transform=None):
        self.images_A = images_A
        self.images_B = images_B
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_A = self.images_A[idx]
        image_B = self.images_B[idx]
        label = self.labels[idx]

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            label = self.transform(label)

        return image_A, image_B, label
