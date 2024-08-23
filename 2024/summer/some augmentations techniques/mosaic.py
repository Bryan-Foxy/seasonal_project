import cv2
import numpy as np
import random
import albumentations as A

class Mosaic(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Mosaic, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return self.mosaic(img)

    def mosaic(self, img):
        h, w, _ = img.shape
        
        # Randomly generate positions for dividing the image
        xc, yc = [int(random.uniform(0.4, 0.6) * s) for s in [w, h]]
        
        # Create a blank canvas for the mosaic
        mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(4):
            # Rescale image randomly
            img_rescaled = cv2.resize(img, (w // 2, h // 2))
            x_off = (i % 2) * (w // 2)
            y_off = (i // 2) * (h // 2)
            
            # Adjust the image size to fit exactly into the mosaic cell
            img_rescaled = img_rescaled[:h // 2, :w // 2, :]
            
            mosaic_img[y_off:y_off + h // 2, x_off:x_off + w // 2] = img_rescaled
        
        return mosaic_img

    def apply_to_bboxes(self, bboxes, **params):
        # Update bounding boxes according to the new mosaic image
        mosaic_bboxes = []
        
        h, w = params['rows'], params['cols']
        xc, yc = [int(random.uniform(0.4, 0.6) * s) for s in [w, h]]
        
        for bbox in bboxes:
            x_min, y_min, width, height = bbox[:4]
            new_bbox = [x_min / 2, y_min / 2, width / 2, height / 2]
            mosaic_bboxes.append(new_bbox)
        
        return mosaic_bboxes

    @property
    def targets(self):
        return {
            "image": self.apply,
            "bboxes": self.apply_to_bboxes,
        }

    def get_params(self):
        return {}