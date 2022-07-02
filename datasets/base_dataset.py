from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale
from datasets.augmentations.geometric_transforms import random_crop, random_hflip, random_scale
from datasets.augmentations.gaussian_blur import GaussianBlur


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.use_augmentation: bool = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def _geometric_augmentations(
            image: Image.Image,
            random_scale_range: Optional[Tuple[float, float]] = None,
            random_crop_size: Optional[int] = None,
            random_hflip_p: Optional[float] = None,
            mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
            ignore_index: Optional[int] = None
    ):
        """Note. image and mask are assumed to be of base size, thus share a spatial shape."""
        if random_scale_range is not None:
            image, mask = random_scale(image=image, random_scale_range=random_scale_range, mask=mask)

        if random_crop_size is not None:
            crop_size = (random_crop_size, random_crop_size)

            fill = tuple(np.array(image).mean(axis=(0, 1)).astype(np.uint8).tolist())
            image, offset = random_crop(image=image, crop_size=crop_size, fill=fill)

            if mask is not None:
                assert ignore_index is not None
                mask = random_crop(image=mask, crop_size=crop_size, fill=ignore_index, offset=offset)[0]

        if random_hflip_p is not None:
            image, mask = random_hflip(image=image, p=random_hflip_p, mask=mask)
        return image, mask

    @staticmethod
    def _photometric_augmentations(
            image: Image.Image,
            random_color_jitter: Optional[Dict[str, float]] = None,
            random_grayscale_p: Optional[float] = None,
            random_gaussian_blur: bool = True
    ):
        if random_color_jitter is None:  # note that "is None" rather than "is not None"
            color_jitter = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            image = RandomApply([color_jitter], p=0.8)(image)

        if random_grayscale_p is not None:
            image = RandomGrayscale(random_grayscale_p)(image)

        if random_gaussian_blur:
            w, h = image.size
            image = GaussianBlur(kernel_size=int((0.1 * min(w, h) // 2 * 2) + 1))(image)
        return image

    def augmentation_on(self) -> None:
        self.use_augmentation = True

    def augmentation_off(self) -> None:
        self.use_augmentation = False
