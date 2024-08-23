import random
import numpy as np
import albumentations as A

class Cutout(A.ImageOnlyTransform):
    def __init__(self, num_holes=8, max_h_size=16, max_w_size=16, always_apply=False, p=0.5):
        super(Cutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def apply(self, image, **params):
        h, w = image.shape[:2]

        for _ in range(self.num_holes):
            # Generate random coordinates for the cutout
            y = random.randint(0, h)
            x = random.randint(0, w)
            # Randomly choose the size of the cutout region
            h_size = random.randint(1, self.max_h_size)
            w_size = random.randint(1, self.max_w_size)

            # Calculate the region to cut out
            y1 = np.clip(y - h_size // 2, 0, h)
            y2 = np.clip(y + h_size // 2, 0, h)
            x1 = np.clip(x - w_size // 2, 0, w)
            x2 = np.clip(x + w_size // 2, 0, w)

            # Apply cutout (fill the region with black pixels)
            image[y1:y2, x1:x2] = 0

        return image

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")