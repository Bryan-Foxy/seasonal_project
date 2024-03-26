import os
import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from config import both_transformations, hr_transformations, lr_transformations

class ImageDataset(torch.utils.data.dataset):
    # Custom Dataset of the image

    def __init__(self, lr_path, hr_path, both_transformations, hr_transformations, lr_transformations):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.both_transformation = both_transformations
        self.hr_transformation = hr_transformations
        self.lr_transformation = lr_transformations
        self.img_lr = []
        self.img_hr = []

        list_img_lr = os.listdir(self.img_lr)
        for i in tqdm(range(len(list_img_lr)), desc = 'Low Resolution loading'):
            self.img_lr.append(os.path.join(self.lr_path, list_img_lr[i]))
        
        list_img_hr = os.listdir(self.img_hr)
        for i in tqdm(range(len(list_img_hr)), desc = 'High Resolution loading'):
            self.img_hr.append(os.path.join(self.hr_path, list_img_hr[i]))

    
    def __len__(self):
        return min(len(self.img_lr), len(self.img_hr))
    
    def __getitem__(self,idx):
        img_lr = cv2.imread(self.img_lr[idx])
        img_hr = cv2.imread(self.img_hr[idx])
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_hr = self.both_transformation(img_hr)
        img_lr = self.both_transformation(img_hr)
        img_hr_tensor = self.hr_transformation
        img_lr_tensor = self.lr_transformation

        img_hr_tensor = torch.permute(img_hr_tensor, (2,0,1))
        img_lr_tensor = torch.permute(img_lr_tensor, (2,0,1))

        return img_hr_tensor, img_lr_tensor
    

def run():
    dataset = ImageDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size = 16)

    for lr, hr in loader:
        print('Low Res: {}'.format(lr.shape))
        print('High Res: {}'.format(hr.shape))

if __name__ == '__main__':
    run()




