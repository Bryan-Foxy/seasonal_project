import os
import cv2
import torch
from tqdm import tqdm

class ImageDataset(torch.utils.data.dataset):
    # Custom Dataset of the image

    def __init__(self, path, transforms, augmentations = False):
        self.path = path
        self.transforms = transforms
        self.augmentations = augmentations
        self.imgs_path = []

        list_img = os.listdir(self.path)
        for i in tqdm(range(len(list_img)), desc = 'Data loading'):
            self.imgs_path.append(os.path.join(self.path, list_img[i]))

    
    def __len__(self):
        return len(self.imgs_path)
    
    def __getitem__(self,idx):
        img = cv2.imread(self.imgs_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img)

        return img_tensor




