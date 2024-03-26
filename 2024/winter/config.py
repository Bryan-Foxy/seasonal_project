import torchvision.transforms as transforms
import torch

# Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
learning_rate = 2e-4
epochs = 5000


hr_w, hr_h = 128,128
lr_w, lr_h = hr_w // 4, hr_h // 4

both_transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((hr_w, hr_h)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation([45,90]),

        ])

hr_transformations = transforms.Compose([
    transforms.Normalize(mean = [0,0,0], std = [1,1,1]),
    transforms.ToTensor(),
])

lr_transformations = transforms.Compose([
    transforms.Resize((lr_w, lr_h)),
    transforms.Normalize(mean = [0,0,0], std = [1,1,1]),
    transforms.ToTensor(),
])