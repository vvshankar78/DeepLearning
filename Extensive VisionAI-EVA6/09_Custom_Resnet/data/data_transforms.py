from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    CoarseDropout,
    RandomCrop,
    CenterCrop,
    PadIfNeeded,
    OneOf,
    ShiftScaleRotate,
    ToGray
)
from albumentations.pytorch import ToTensorV2
import numpy as np
from cv2 import BORDER_CONSTANT, BORDER_REFLECT

def albumentations_transforms(p=1.0, is_train=False):
	# Mean and standard deviation of test+train dataset
    # (0.4919, 0.4827, 0.4472), (0.2470, 0.2434, 0.2616)
	mean = np.array([0.4919, 0.4827, 0.4472])
	std = np.array([0.2470, 0.2434, 0.2616])
	transforms_list = []
	# Use data aug only for train data
	if is_train:
		transforms_list.extend([
            # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            PadIfNeeded(min_height=38, min_width=38, border_mode=BORDER_CONSTANT, value=mean*255.0, p=1.0),
			OneOf([
				RandomCrop(height=38, width=38, p=0.8),
				HorizontalFlip(p=0.5),
				#CenterCrop(height=32, width=32, p=0.2),
                CoarseDropout(max_holes=1, min_holes=1, max_height=8, max_width=8, min_height=8,
				min_width=8, fill_value=mean*255.0, mask_fill_value=None, p=0.7),
                # ToGray(p=0.3),
			], p=1.0),])
    
	if is_train == False:
		transforms_list.extend([
			# HorizontalFlip(p=0.5),
            # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            PadIfNeeded(min_height=38, min_width=38, border_mode=BORDER_CONSTANT, value=mean*255.0, p=1.0)])


    
	transforms_list.extend([
		Normalize(
			mean=mean,
			std=std,
			max_pixel_value=255.0,
			p=1.0
		),
		ToTensorV2()
	])
	transforms = Compose(transforms_list, p=p)
	return lambda img:transforms(image=np.array(img))["image"]
