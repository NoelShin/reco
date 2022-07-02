import os
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import normalize, to_tensor

from torch.utils.data import Dataset
from PIL import Image


class PascalContextDataset(Dataset):
    def __init__(
            self,
            dir_dataset: str,
            split: str = "val",
            size: int = 320,
            model_name: str = "RN50",
    ):
        super(PascalContextDataset, self).__init__()
        assert os.path.exists(dir_dataset), FileNotFoundError(f"{dir_dataset}")

        self.dir_dataset = dir_dataset
        self.category: Optional[str] = None
        self.label_id: Optional[int] = None
        self.split: str = split
        self.model_name: str = model_name
        self.size: int = size

        self.mean: List[float, float, float] = [0.485, 0.456, 0.406]
        self.std: List[float, float, float] = [0.229, 0.224, 0.225]

        self.p_imgs: List[str] = self._get_image_paths(split=split)
        self.p_gts: List[str] = self._get_ground_truth_paths(split=split)
        self.n_categories: int = 59

    @staticmethod
    def read_txt_file(p_txt_file) -> List[str]:
        with open(p_txt_file, 'r') as f:
            list_lines = f.readlines()
            f.close()
        list_lines = [l.replace('\n', '') for l in list_lines]
        return list_lines

    def _get_image_paths(self, split: str) -> List[str]:
        assert split in ["train", "val"]
        img_filenames: List[str] = self.read_txt_file(f"{self.dir_dataset}/ImageSets/SegmentationContext/{split}.txt")
        return [f"{self.dir_dataset}/JPEGImages/{filename}.jpg" for filename in img_filenames]

    def _get_ground_truth_paths(self, split: str) -> List[str]:
        assert split in ["train", "val"]
        img_filenames: List[str] = self.read_txt_file(f"{self.dir_dataset}/ImageSets/SegmentationContext/{split}.txt")
        return [f"{self.dir_dataset}/SegmentationClassContext/{filename}.png" for filename in img_filenames]

    def __len__(self):
        return len(self.p_imgs)

    def __getitem__(self, ind: int) -> Dict[str, Union[str, torch.Tensor]]:
        dict_data: Dict[str, Union[str, torch.Tensor]] = dict()

        p_img: str = self.p_imgs[ind]
        p_gt: str = self.p_gts[ind]

        img: Image.Image = Image.open(p_img).convert("RGB")
        img = TF.center_crop(TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR), self.size)
        img: torch.Tensor = normalize(to_tensor(img), mean=self.mean, std=self.std)

        gt: Image.Image = Image.open(p_gt)
        gt = TF.center_crop(TF.resize(gt, self.size, interpolation=TF.InterpolationMode.NEAREST), self.size)
        gt: torch.LongTensor = torch.from_numpy(np.array(gt)).to(torch.int64)
        gt = gt - 1  # shift all label ids by minus one such that the background regions will be -1

        dict_data.update({
            "img": img,
            "gt": gt,
            "p_img": p_img,
            "p_gt": p_gt
        })
        return dict_data


pascal_context_categories = [
    'aeroplane',
    'bag',
    'bed',
    'bedclothes',
    'bench',
    'bicycle',
    'bird',
    'boat',
    'book',
    'bottle',
    'building',
    'bus',
    'cabinet',
    'car',
    'cat',
    'ceiling',
    'chair',
    'cloth',
    'computer',
    'cow',
    'cup',
    'curtain',
    'dog',
    'door',
    'fence',
    'floor',
    'flower',
    'food',
    'grass',
    'ground',
    'horse',
    'keyboard',
    'light',
    'motorbike',
    'mountain',
    'mouse',
    'person',
    'plate',
    'platform',
    'pottedplant',
    'road',
    'rock',
    'sheep',
    'shelves',
    'sidewalk',
    'sign',
    'sky',
    'snow',
    'sofa',
    'table',
    'track',
    'train',
    'tree',
    'truck',
    'tvmonitor',
    'wall',
    'water',
    'window',
    'wood'
]

pascal_context_pallete = [
    [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
           [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
           [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140],
           [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
           [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
           [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
           [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
           [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8],
           [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8],
           [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
           [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
           [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
           [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0],
           [0, 235, 255], [0, 173, 255], [31, 0, 255]
]
