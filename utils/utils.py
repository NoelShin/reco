from typing import Optional
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image


def get_model(
        model_name: str,
        patch_size: Optional[int] = 16,
        denseclip: bool = True,
        device=torch.device("cuda:0")
):
    if model_name in ["mocov2", "swav", "supervised"]:
        from networks.resnet.resnet import ResNet50
        model = ResNet50(model_name, use_dilated_resnet=False).to(device)

    elif model_name == "dino_small":
        from networks.vit.vision_transformer import deit_small, load_model
        model = deit_small(patch_size=patch_size).to(device)
        load_model(model, arch="deit_small", patch_size=patch_size)
        model_name = f"{model_name}_p{patch_size}"

    elif model_name == "dino_base":
        from networks.vit.vision_transformer import vit_base, load_model
        model = vit_base(patch_size=patch_size).to(device)
        load_model(model, arch="vit_base", patch_size=patch_size)
        model_name = f"{model_name}_p{patch_size}"

    elif model_name == "DeiT-S/16-SIN":
        from networks.vit.distilled_vision_transformer import dino_small_dist
        model = dino_small_dist(patch_size=16, pretrained=True).to(device)
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth",
            map_location="cuda:0"
        )
        model.load_state_dict(state_dict["model"], strict=False)
        model_name = f"{model_name.replace('/', '_').replace('-', '_').lower()}"

    else:
        # load a CLIP or DenseCLIP model
        if "RN" in model_name and denseclip:
            # DenseCLIP
            from networks.dense_clip import DenseCLIP
            model = DenseCLIP(arch_name=model_name).to(device)
        else:
            # CLIP
            import clip
            from networks.clip_arch import build_model
            model, transforms = clip.load(model_name)
            model = build_model(model.state_dict()).to(device)
            model_name = f"{model_name.replace('/', '_').replace('-', '_').replace('@', '_').lower()}"
    model.requires_grad_(False)
    model.eval()
    print(f"{model_name} is successfully loaded.")
    return model


# copied from https://github.com/mhamilton723/STEGO/blob/master/src/data.py
class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)

# copied from https://github.com/mhamilton723/STEGO/blob/master/src/data.py
def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([
            T.Resize(res, Image.NEAREST),
            cropper,
            ToTargetTensor()
        ])
    else:
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return T.Compose([
            T.Resize(res, Image.NEAREST),
            cropper,
            T.ToTensor(),
            normalize
        ])


def get_dataset(
        dir_dataset: str,
        dataset_name: str,
        split: str = "val",
        dense_clip_arch: Optional[str] = None,
        dir_pseudo_masks: Optional[str] = None
):
    if dataset_name == "cityscapes":
        from datasets.cityscapes import CityscapesDataset, cityscapes_categories, cityscapes_pallete

        loader_crop = "center"
        dataset = CityscapesDataset(
            dir_dataset=dir_dataset,
            split=split,
            transform=get_transform(res=320, is_label=False, crop_type=loader_crop),
            target_transform=get_transform(res=320, is_label=True, crop_type=loader_crop),
            model_name=dense_clip_arch,
            dir_pseudo_masks=dir_pseudo_masks
        )
        categories = cityscapes_categories
        pallete = cityscapes_pallete

    elif dataset_name == "coco_stuff":
        from datasets.coco_stuff import COCOStuffDataset, coco_stuff_pallete, coco_stuff_categories_fine

        dataset = COCOStuffDataset(
            dir_dataset=dir_dataset,
            split=f"train10k" if split == "train" else split,
            transform=get_transform(res=320, is_label=False, crop_type="center"),
            target_transform=get_transform(res=320, is_label=True, crop_type="center"),
            coarse_labels=False,
            exclude_things=False,
            model_name=dense_clip_arch,
            dir_pseudo_masks=dir_pseudo_masks
        )
        categories = coco_stuff_categories_fine  # coco_stuff_categories
        pallete = coco_stuff_pallete

    elif dataset_name == "pascal_context":
        from datasets.pascal_context import pascal_context_categories, PascalContextDataset, pascal_context_pallete

        dataset = PascalContextDataset(
            dir_dataset=dir_dataset,
            split=split,
            model_name=dense_clip_arch,
        )

        categories = pascal_context_categories
        pallete = pascal_context_pallete

    elif dataset_name == "kitti_step":
        from datasets.kitti_step import KittiStepDataset, kitti_step_categories, kitti_step_palette

        dataset = KittiStepDataset(
            dir_dataset=dir_dataset,
            split=split,
            model_name=dense_clip_arch,
            dir_pseudo_masks=dir_pseudo_masks
        )
        categories = kitti_step_categories
        pallete = kitti_step_palette

    else:
        raise ValueError(dataset_name)
    return dataset, categories, pallete


voc_palette = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128],
    7: [128, 128, 128],
    8: [64, 0, 0],
    9: [192, 0, 0],
    10: [64, 128, 0],
    11: [192, 128, 0],
    12: [64, 0, 128],
    13: [192, 0, 128],
    14: [64, 128, 128],
    15: [192, 128, 128],
    16: [0, 64, 0],
    17: [128, 64, 0],
    18: [0, 192, 0],
    19: [128, 192, 0],
    20: [0, 64, 128],
    255: [255, 255, 255]
}


def colourise_mask(
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        benchmark: str = "voc",
        opacity: float = 0.5
):
    # assert label.dtype == np.uint8
    assert len(mask.shape) == 2, ValueError(mask.shape)
    h, w = mask.shape
    grid = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = set(mask.flatten())
    if "voc" in benchmark:
        palette = list(voc_palette.values())

        # coloured_label = label2rgb(label, image=image, colors=palette, alpha=opacity)
    else:
        raise ValueError(benchmark)

    for l in unique_labels:
        try:
            grid[mask == l] = np.array(palette[l])
        except IndexError:
            print(l)

    return grid


# copied from https://github.com/mhamilton723/STEGO/blob/master/src/utils.py
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])