# modifed based on https://github.com/mhamilton723/STEGO/blob/master/src/data.py
import os
from glob import glob
import random
from typing import Optional, List
import numpy as np
import torch.multiprocessing
from PIL import Image
from torchvision.datasets.cityscapes import Cityscapes
import torchvision.transforms.functional as TF
from datasets.base_dataset import BaseDataset


class CityscapesDataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: str,
            split: str,
            transform,
            target_transform,
            model_name: str = "RN50x16",
            dir_pseudo_masks: Optional[str] = None
    ):
        super(CityscapesDataset, self).__init__()
        self.split = split
        self.dir_dataset = dir_dataset
        if split == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = split
            mode = "fine"
        self.inner_loader = Cityscapes(
            self.dir_dataset,
            our_image_set,
            mode=mode,
            target_type="semantic",
            transform=None,
            target_transform=None
        )
        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7

        # noel: img path
        self.p_imgs: List[str] = sorted(glob(f"{dir_dataset}/leftImg8bit/{split}/**/*.png"))
        self.p_gts: List[str] = sorted(glob(f"{dir_dataset}/gtFine/{split}/**/*_gtFine_labelIds.png"))

        self.model_name: str = model_name
        self.n_categories: int = 27
        self.use_augmentation = False
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.name: str = "cityscapes"

        if dir_pseudo_masks is not None:
            assert os.path.exists(dir_pseudo_masks)
            self.p_pseudo_gts: List[str] = sorted(glob(f"{dir_pseudo_masks}/*_gtFine_labelIds.png"))
            assert len(self.p_pseudo_gts) > 0
        else:
            self.p_pseudo_gts: list = []

    def __getitem__(self, index):
        if self.use_augmentation:
            image, _ = self.inner_loader[index]
            p_pseudo_gt = self.p_pseudo_gts[index]
            pseudo_gt = Image.open(p_pseudo_gt)

            image = TF.resize(image, 320)
            image = TF.center_crop(image, 320)

            image, pseudo_gt = self._geometric_augmentations(
                image=image,
                mask=pseudo_gt,
                random_scale_range=(0.5, 2.0),
                random_crop_size=320,
                ignore_index=255,
                random_hflip_p=0.5
            )
            image: Image.Image = self._photometric_augmentations(image)

            image = TF.normalize(TF.to_tensor(image), self.mean, self.std)
            return {
                "img": image,
                "p_img": self.p_imgs[index],
                "p_gt": self.p_gts[index],
                "pseudo_gt": torch.tensor(np.asarray(pseudo_gt, np.int64), dtype=torch.long)
            }

        if self.transform is not None:
            image, target = self.inner_loader[index]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)

            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1

            if len(self.p_pseudo_gts) > 0:
                p_pseudo_gt = self.p_pseudo_gts[index]
                pseudo_gt = Image.open(p_pseudo_gt)
                pseudo_gt = self.target_transform(pseudo_gt).squeeze(dim=0)
            else:
                pseudo_gt = torch.tensor(0.)

            return {
                "img": image,
                "p_img": self.p_imgs[index],
                "gt": target.squeeze(0),
                "p_gt": self.p_gts[index],
                "pseudo_gt": pseudo_gt,
                "void_mask": mask
            }
        else:
            return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)

# for the full label info: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
cityscapes_categories = [
    "road",
    "sidewalk",
    "parking lot",  #parking lot
    "rail track",
    "building",
    "wall",
    "fence",
    "guard rail",
    "bridge",
    "tunnel",
    "pole",
    "polegroup",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",  # "grass",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "train",
    "motorcycle",
    "bicycle"
]

cat_to_label_id = {cat: i for i, cat in enumerate(cityscapes_categories)}

cityscapes_pallete = [
    (128, 64, 128),
    (244, 35, 232),
    (250,170,160),
    (230,150,140),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (180,165,180),
    (150,100,100),
    (150,120, 90),
    (153,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0,  0, 90),
    (  0,  0,110),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32)
]
