import os
from glob import glob
from typing import Dict, List, Optional
import pickle as pkl
from time import time
import numpy as np
import torch
from torchvision.transforms.functional import center_crop, normalize, resize, to_tensor
from torch.utils.data import Dataset
from PIL import Image
from utils.utils import get_model
from utils.extract_text_embeddings import prompt_engineering


class ImageNet1KDataset(Dataset):
    def __init__(
            self,
            dir_dataset: str,
            category: Optional[str] = None,
            k: int = 50,
            size: int = 224,
            split: str = "val",
            model_name: str = "ViT-L/14@336px",
            device: torch.device = torch.device("cuda:0")
    ):
        super(ImageNet1KDataset, self).__init__()
        assert os.path.exists(dir_dataset), FileNotFoundError(f"{dir_dataset}")
        assert split in ["train", "val"], ValueError(split)
        assert model_name in ["RN50", "RN50x16", "RN50x64", "ViT-L/14@336px"], ValueError(model_name)

        self.dir_dataset: str = dir_dataset

        self.k: int = k
        self.size: int = size
        self.split: str = split

        self.device: torch.device = device
        self.model_name: str = model_name
        self.clip = get_model(model_name, denseclip=False)

        p_filename_to_img_embedding = f"{dir_dataset}/filename_to_{model_name.replace('-', '_').replace('/', '_').replace('@', '_')}_{split}_img_embedding.pkl"
        if not os.path.exists(p_filename_to_img_embedding):
            import subprocess
            print("(Retrieval) Extracting img feature for each image in ImageNet1K...")
            print(f"The resulting file will be saved at {p_filename_to_img_embedding}.")
            subprocess.call(
                # f"python3 /users/gyungin/reco/utils/extract_image_embeddings.py -dd {dir_dataset} -m {model_name} -s {split}".split(' ')
                f"python3 ../utils/extract_image_embeddings.py -dd {dir_dataset} -m {model_name} -s {split}".split(' ')
            )
        st = time()
        self.filename_to_img_embedding: Dict[str, torch.Tensor] = pkl.load(
            open(p_filename_to_img_embedding, "rb")
        )
        print(
            f"(Retrieval) A filename-to-img-feature file is loaded from {p_filename_to_img_embedding} "
            f"({time() - st:.3f} sec.)."
        )

        self.filenames: List[str] = list(self.filename_to_img_embedding.keys())  # n_imgs
        self.img_embedding: torch.Tensor = torch.stack(
            list(self.filename_to_img_embedding.values()), dim=0
        ).to(device)  # n_imgs x n_dims

        self.mean: List[float, float, float] = [0.485, 0.456, 0.406]
        self.std: List[float, float, float] = [0.229, 0.224, 0.225]

        self.p_ret_imgs: Optional[List[str]] = None

        if category is not None:
            assert isinstance(category, str), TypeError(f"{type(category)} != str")
            self.set_category(category=category)

    def set_category(
            self,
            category: str,
            verbose: bool = True
    ) -> None:
        category_text_feature = prompt_engineering(model_name=self.clip, categories=[category])[category]
        predictions: torch.FloatTensor = category_text_feature @ self.img_embedding.t()  # 1 x n_imgs

        indices: torch.Tensor = torch.argsort(predictions.squeeze(dim=0), descending=True)
        sorted_filenames: List[str] = np.array(self.filenames)[indices.cpu().tolist()].tolist()
        ret_filenames: List[str] = sorted_filenames[:self.k]  # topk retrieved images
        self.p_ret_imgs: List[str] = list()
        if self.split == "val":
            p_imgs: List[str] = sorted(glob(f"{self.dir_dataset}/val/**/*.JPEG"))
            filename_to_p_img: Dict[str, str] = dict()
            for p_img in p_imgs:
                filename = os.path.basename(p_img)
                filename_to_p_img[filename] = p_img

            for filename in ret_filenames:
                p_img = filename_to_p_img[filename]
                self.p_ret_imgs.append(p_img)
        else:
            for filename in ret_filenames:
                wnid: str = filename.split('_')[0]
                p_img: str = f"{self.dir_dataset}/{wnid}/{filename}"
                self.p_ret_imgs.append(p_img)
        assert len(self.p_ret_imgs) > 0, ValueError(f"{len(self.p_ret_imgs)} == 0.")

        if verbose:
            print(f"Category has been set to {category} (ImageNet1K).")

    def __len__(self):
        return len(self.p_ret_imgs)

    def __getitem__(self, ind):
        imgs: List[torch.Tensor] = list()
        for p_img in self.p_ret_imgs:
            if self.split == "train":
                img: Image.Image = Image.open(p_img.replace('Net2012', 'Net2012/train')).convert("RGB")
            else:
                img: Image.Image = Image.open(p_img).convert("RGB")

            img: Image.Image = resize(img, size=self.size)
            # clip also center-crops input images and the kNN was done with center-cropped images
            img: Image.Image = center_crop(img, output_size=self.size)
            img: torch.Tensor = normalize(to_tensor(img), mean=self.mean, std=self.std)
            imgs.append(img)

        dict_data = {
            "imgs": torch.stack(imgs, dim=0),
            "p_imgs": self.p_ret_imgs
        }
        return dict_data