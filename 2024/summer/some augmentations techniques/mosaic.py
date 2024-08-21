"""
The idea behind Mosaic is very simple. 
Take 4 images and combine them into a single image. 
Mosaic does this by resizing each of the four images, stitching them together, and then taking a random cutout of the stitched images to get the final Mosaic image.
"""
import cv2
import torch 

class Mosaic_augmentation:
    def __init__(self, images, bboxes):
        super(Mosaic_augmentation, self).__init__()
        self.images = images
        self.bboxes = bboxes
        self.mosaic_images = torch.zeros((self.images.shape, ))
    
    def augment(self, resize):
        return 0