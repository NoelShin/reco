# modifed based on https://github.com/mhamilton723/STEGO/blob/master/src/data.py
import os
from glob import glob
import random
from typing import Optional, List
import numpy as np
import scipy.io
import torch.multiprocessing
from PIL import Image
import torchvision.transforms.functional as TF
from datasets.base_dataset import BaseDataset


class COCOStuffDataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: str,
            # root: str
            split: str,
            transform,
            target_transform,
            coarse_labels,
            exclude_things,
            subset: Optional[int] = None,
            model_name: str = "RN50",
            dir_pseudo_masks: Optional[str] = None
    ):
        super(COCOStuffDataset, self).__init__()
        self.split = split
        self.dir_dataset = dir_dataset
        # self.root = os.path.join(root, "cocostuff")
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val", "train+val", "train10k", "val10k"]
        # noel - note that val10k won't be used in the paper.
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        if "10k" in split:
            # noel
            with open(f"{self.dir_dataset}10k/imageLists/{split.replace('10k', '')}.txt", "r") as f:
                lines = f.readlines()
                filenames = [line.replace('\n', '') for line in lines]
                self.image_files = [f"{self.dir_dataset}10k/images/{fn}.jpg" for fn in filenames]
                self.label_files = [f"{self.dir_dataset}10k/annotations/{fn}.mat" for fn in filenames]
                f.close()

        else:
            self.image_files = []
            self.label_files = []
            for split_dir in split_dirs[self.split]:
                with open(os.path.join(self.dir_dataset, "curated", split_dir, self.image_list), "r") as f:
                    img_ids = [fn.rstrip() for fn in f.readlines()]
                    for img_id in img_ids:
                        self.image_files.append(os.path.join(self.dir_dataset, "images", split_dir, img_id + ".jpg"))
                        self.label_files.append(os.path.join(self.dir_dataset, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self._label_names = [
            "ground-stuff",
            "plant-stuff",
            "sky-stuff",
        ]
        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

        self.n_categories: int = 27
        self.use_augmentation = False
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.name: str = "coco_stuff"

        if dir_pseudo_masks is not None:
            assert os.path.exists(dir_pseudo_masks), FileNotFoundError(dir_pseudo_masks)
            self.p_pseudo_gts: List[str] = sorted(glob(f"{dir_pseudo_masks}/*.png"))
        else:
            self.p_pseudo_gts: list = []

    def __getitem__(self, index):
        if self.use_augmentation:
            image_path = self.image_files[index]
            image = Image.open(image_path).convert("RGB")  # H x W x 3

            p_pseudo_gt = self.p_pseudo_gts[index]
            pseudo_gt = Image.open(p_pseudo_gt)

            image = TF.center_crop(TF.resize(image, 320), 320)
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
                "p_img": image_path,
                "p_gt": p_pseudo_gt,
                "pseudo_gt": torch.tensor(np.asarray(pseudo_gt, np.int64), dtype=torch.long)
            }

        else:
            image_path = self.image_files[index]
            label_path = self.label_files[index]
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform(Image.open(image_path).convert("RGB"))

            random.seed(seed)
            torch.manual_seed(seed)
            if os.path.splitext(label_path)[-1] == ".mat":
                # keys: 'S', "captions", "names", "regionLabelsStuff", "regionMapStuff"
                mat: dict = scipy.io.loadmat(label_path)
                label: np.ndarray = mat["S"]
                label = label - 1  # shift by 1 s.t. the background is -1
                label = Image.fromarray(label)
            else:
                label = Image.open(label_path)

            label = self.label_transform(label).squeeze(0)
            label[label == 255] = -1  # to be consistent with 10k

            coarse_label = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                coarse_label[label == fine] = coarse
            coarse_label[label == -1] = -1

            if self.coarse_labels:
                coarser_labels = -torch.ones_like(label)
                for i, c in enumerate(self.cocostuff3_coarse_classes):
                    coarser_labels[coarse_label == c] = i
                return img, coarser_labels, coarser_labels >= 0
            else:
                if self.exclude_things:
                    return img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
                else:
                    return {"img": img, "p_img": image_path, "gt": coarse_label, "p_gt": label_path}

    def __len__(self):
        return len(self.image_files)


def create_pascal_label_colormap() -> np.ndarray:
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    def bit_get(val, idx):
        """Gets the bit value.
        Args:
          val: Input value, int or numpy int array.
          idx: Which bit of the input val.
        Returns:
          The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap

coco_stuff_pallete = create_pascal_label_colormap()  # 512 x 3

# copied from https://github.com/xu-ji/IIC/blob/master/code/datasets/segmentation/util/cocostuff_fine_to_coarse.py
# first 12 are "things", next 15 are "stuff"
# arbitrary order as long as things first, then stuff
# For the details about the two food categories below, check
# https://raw.githubusercontent.com/nightrome/cocostuff10k/master/dataset/cocostuff-labelhierarchy.png
coco_stuff_categories = [
  "electronic",  # 0
  "appliance",  # 1
  "food things",  # 2, food-things, i.e., cake, donut, pizza, hot dog, carrot, broccoli, orange, sandwich, apple, and banana
  "furniture things",  # 3, furniture-thing
  "indoor",  # 4
  "kitchen",  # 5
  "accessory",  # 6
  "animal",  # 7
  "outdoor",  # 8
  "person",  # 9
  "sports",  # 10
  "vehicle",  # 11

  "ceiling",  # 12
  "floor",  # 13
  "food stuff",  # 14, food-stuff, i.e., food-other, vegetable, salad, and fruit
  "furniture stuff",  # 15
  "raw material",  # 16
  "textile",  # 17
  "wall",  # 18
  "window",  # 19
  "building",  # 20
  "ground",  # 21
  "plant",  # 22
  "sky",  # 23
  "solid",  # 24
  "structural",  # 25
  "water"  # 26
]


label_id_to_category_fine = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    # 12: "street sign", removed from COCO
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    # 26: "hat", removed from COCO
    27: "backpack",
    28: "umbrella",
    # 29: "shoe", removed from COCO
    # 30: "eye glasses", removed from COCO
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    # 45: "plate", removed from COCO
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    # 66: "mirror", removed from COCO
    67: "dining table",
    # 68: "window", removed from COCO
    # 69: "desk", removed from COCO
    70: "toilet",
    # 71: "door", removed from COCO
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    # 83: "blender", removed from COCO
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    # 91: "hair brush", removed from COCO
    92: "banner",
    93: "blanket",
    94: "branch",
    95: "bridge",
    96: "building other",
    97: "bush",
    98: "cabinet",
    99: "cage",
    100: "cardboard",
    101: "carpet",
    102: "ceiling other",
    103: "ceiling tile",
    104: "cloth",
    105: "clothes",
    106: "clouds",
    107: "counter",
    108: "cupboard",
    109: "curtain",
    110: "desk stuff",
    111: "dirt",
    112: "door stuff",
    113: "fence",
    114: "floor marble",
    115: "floor other",
    116: "floor stone",
    117: "floor tile",
    118: "floor wood",
    119: "flower",
    120: "fog",
    121: "food other",
    122: "fruit",
    123: "furniture other",
    124: "grass",
    125: "gravel",
    126: "ground other",
    127: "hill",
    128: "house",
    129: "leaves",
    130: "light",
    131: "mat",
    132: "metal",
    133: "mirror stuff",
    134: "moss",
    135: "mountain",
    136: "mud",
    137: "napkin",
    138: "net",
    139: "paper",
    140: "pavement",
    141: "pillow",
    142: "plant other",
    143: "plastic",
    144: "platform",
    145: "playingfield",
    146: "railing",
    147: "railroad",
    148: "river",
    149: "road",
    150: "rock",
    151: "roof",
    152: "rug",
    153: "salad",
    154: "sand",
    155: "sea",
    156: "shelf",
    157: "sky other",
    158: "skyscraper",
    159: "snow",
    160: "solid other",
    161: "stairs",
    162: "stone",
    163: "straw",
    164: "structural other",
    165: "table",
    166: "tent",
    167: "textile other",
    168: "towel",
    169: "tree",
    170: "vegetable",
    171: "wall brick",
    172: "wall concrete",
    173: "wall other",
    174: "wall panel",
    175: "wall stone",
    176: "wall tile",
    177: "wall wood",
    178: "water other",
    179: "waterdrops",
    180: "window blind",
    181: "window other",
    182: "wood"
}

coco_stuff_182_to_27 = {
    0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
    13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
    25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
    37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
    49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
    61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
    73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
    85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
    97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
    107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
    117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
    127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
    137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
    147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
    157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
    167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
    177: 26, 178: 26, 179: 19, 180: 19, 181: 24
}

coco_stuff_182_to_171: dict = {}
cnt: int = 0
for label_id in coco_stuff_182_to_27:
    if label_id in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
        continue
    coco_stuff_182_to_171[label_id] = cnt
    cnt += 1

coco_stuff_171_to_27 = dict()
cnt = 0
for fine, coarse in coco_stuff_182_to_27.items():
    if fine in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
        continue
    coco_stuff_171_to_27[cnt] = coarse
    cnt += 1

coco_stuff_categories_fine = list(label_id_to_category_fine.values())[1:]  # exclude background



