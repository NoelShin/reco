# modifed based on https://github.com/mhamilton723/STEGO/blob/master/src/data.py
import os
from glob import glob
from typing import Dict, Optional, Union, List
import numpy as np
import torch.multiprocessing
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import normalize, to_tensor
from datasets.base_dataset import BaseDataset


class KittiStepDataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: str,
            split: str,
            model_name: str = "RN50x16",
            dir_pseudo_masks: Optional[str] = None
    ):
        super(KittiStepDataset, self).__init__()
        assert os.path.exists(f"{dir_dataset}")
        self.split = split
        self.dir_dataset = dir_dataset
        if split == "train":
            # 5,027 images
            # note: label ids of {1, 12, 16, 17} are not present in the validation masks.
            self.p_imgs: List[str] = sorted(glob(f"{dir_dataset}/train/**/*.png"))
            self.p_gts: List[str] = sorted(glob(f"{dir_dataset}/panoptic_maps/train/**/*.png"))

        else:
            # 2,981 images
            self.p_imgs: List[str] = sorted(glob(f"{dir_dataset}/val/**/*.png"))
            self.p_gts: List[str] = sorted(glob(f"{dir_dataset}/panoptic_maps/val/**/*.png"))
        assert len(self.p_imgs) == len(self.p_gts), ValueError(f"# imgs ({self.p_imgs}) != # gts ({self.p_gts})")

        self.model_name: str = model_name
        self.n_categories: int = 19
        self.use_augmentation = False
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.name: str = "kitti_step"

        if dir_pseudo_masks is not None:
            assert os.path.exists(dir_pseudo_masks), FileNotFoundError(dir_pseudo_masks)
            self.p_pseudo_gts: List[str] = sorted(glob(f"{dir_pseudo_masks}/**/*.png"))
            assert len(self.p_pseudo_gts) > 0
            assert len(self.p_pseudo_gts) == len(self.p_imgs)
            for p_gt, p_pseudo_gt in zip(self.p_gts, self.p_pseudo_gts):
                assert p_gt.split('/')[-1] == p_pseudo_gt.split('/')[-1]
        else:
            self.p_pseudo_gts: list = []

    def __getitem__(self, index):
        if self.use_augmentation:
            p_img: str = self.p_imgs[index]
            p_pseudo_gt = self.p_pseudo_gts[index]

            image: Image.Image = Image.open(p_img).convert("RGB")
            pseudo_gt: Image.Image = Image.open(p_pseudo_gt)

            # for the consistency with the Cityscapes and COCO-Stuff, we resize and center-crop with 320x320 for training
            image: Image.Image = TF.resize(image, 320)
            image: Image.Image = TF.center_crop(image, 320)

            pseudo_gt: Image.Image = TF.resize(pseudo_gt, 320,  interpolation=TF.InterpolationMode.NEAREST)
            pseudo_gt: Image.Image = TF.center_crop(pseudo_gt, 320)

            image, pseudo_gt = self._geometric_augmentations(
                image=image,
                mask=pseudo_gt,
                random_scale_range=(0.5, 2.0),
                random_crop_size=320,
                ignore_index=255,
                random_hflip_p=0.5
            )
            image: Image.Image = self._photometric_augmentations(image)
            image: torch.Tensor = TF.normalize(TF.to_tensor(image), self.mean, self.std)

            return {
                "img": image,
                "p_img": self.p_imgs[index],
                "p_gt": self.p_gts[index],
                "pseudo_gt": torch.tensor(np.asarray(pseudo_gt, np.int64), dtype=torch.long)
            }

        else:
            dict_data: Dict[str, Union[str, torch.Tensor]] = dict()

            p_img: str = self.p_imgs[index]
            p_gt: str = self.p_gts[index]

            img: Image.Image = Image.open(p_img).convert("RGB")
            img: torch.Tensor = normalize(to_tensor(img), mean=self.mean, std=self.std)

            gt: Image.Image = Image.open(p_gt).convert("RGB")
            gt: torch.LongTensor = torch.tensor(np.array(gt, dtype=np.int64))[..., 0]
            gt[gt == 255] = -1  # change the background label (255) to -1

            dict_data.update({
                "img": img,
                "gt": gt,
                "p_img": p_img,
                "p_gt": p_gt
            })

            return dict_data

    def __len__(self):
        return len(self.p_imgs)

# for the full label info: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
kitti_step_categories = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
]

cat_to_label_id = {cat: i for i, cat in enumerate(kitti_step_categories)}

palette_cs = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32)
}

kitti_step_palette = [v for v in palette_cs.values()]
