# Setup
import os
import cv2
import torch
from torch.utils.data import dataloader, dataset
from esrgan import Generator, Discriminator

disc_model = Discriminator()
gen_model = Generator()



