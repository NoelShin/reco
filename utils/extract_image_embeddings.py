import os
from typing import Dict, List, Optional
from math import sqrt
import pickle as pkl
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import clip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


@torch.no_grad()
def extract_image_embeddings(
        p_imgs: Optional[List[str]] = None,
        dataloader: Optional[DataLoader] = None,
        model_name: str = "RN50",
        fp: Optional[str] = None,
        size: Optional[int] = None,
        device: torch.device = torch.device("cuda:0"),
) -> Dict[str, torch.FloatTensor]:
    assert p_imgs is not None or dataloader is not None
    assert model_name in ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],\
        ValueError(model_name)
    print(f"Extracting features using {model_name}...")
    model, preprocess = clip.load(model_name, device=device)
    if size is not None and "RN" in model_name:
        # only for resnet architectures
        pos_emb = model.visual.attnpool.positional_embedding
        cls_token, patch_tokens = pos_emb[0], pos_emb[1:]

        h_feat = w_feat = int(sqrt(len(patch_tokens)))
        h_feat_new = w_feat_new = int(size // 32)  # 32 is the total stride of CLIP visual encoder

        # h_feat x w_feat x n_dims -> n_dims x h_feat x w_feat
        patch_tokens = patch_tokens.view(h_feat, w_feat, -1).permute(2, 0, 1)
        resized_patch_tokens = F.interpolate(
            patch_tokens[None],
            size=(h_feat_new, w_feat_new),
            mode="bicubic",
            align_corners=True
        )[0].view(-1, h_feat_new * w_feat_new).permute(1, 0)  # h_feat_new * w_feat_new x n_dims
        model.visual.attnpool.positional_embedding = nn.Parameter(
            torch.cat([cls_token[None], resized_patch_tokens], dim=0)
        )

    filename_to_img_embedding: Dict[str, torch.FloatTensor] = dict()

    if dataloader is not None:
        n_total_iters = len(dataloader)
        for num_iter, dict_data in enumerate(tqdm(dataloader)):
            imgs, p_imgs = dict_data["img"], dict_data["p_img"]
            img_embeddings: torch.FloatTensor = model.encode_image(imgs.to(device))  # b x 1024
            img_embeddings = img_embeddings / torch.linalg.norm(img_embeddings, ord=2, dim=1, keepdim=True)  # b x 1024
            img_embeddings = img_embeddings.cpu()

            for i, p_img in enumerate(p_imgs):
                # float16 -> float32
                filename_to_img_embedding[os.path.basename(p_img)] = img_embeddings[i].to(torch.float32)

            if num_iter % (n_total_iters // 20) == 0 and fp is not None:
                pkl.dump(filename_to_img_embedding, open(fp, "wb"))

    else:
        for p_img in tqdm(p_imgs):
            img = preprocess(Image.open(p_img)).unsqueeze(0).to(device)
            img_embedding: torch.FloatTensor = model.encode_image(img).squeeze(dim=0)  # 1 x 1024 -> 1024
            img_embedding = img_embedding / torch.linalg.norm(img_embedding, ord=2, keepdim=True)
            filename_to_img_embedding[os.path.basename(p_img)] = img_embedding.to(torch.float32)  # float16 -> float32

    if fp is not None:
        pkl.dump(filename_to_img_embedding, open(fp, "wb"))
    return filename_to_img_embedding


class ImageNet1KDataset(Dataset):
    def __init__(
            self,
            dir_dataset: str,
            split: str,
            size: int
    ):
        self.p_imgs: List[str] = sorted(glob(f"{dir_dataset}/{split}/**/*.JPEG"))
        self.transforms = Compose([
            Resize(size, interpolation=BICUBIC),
            CenterCrop(size),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def __len__(self):
        return len(self.p_imgs)

    def __getitem__(self, ind):
        return {
            "img": self.transforms(Image.open(self.p_imgs[ind])),
            "p_img": self.p_imgs[ind]
        }


if __name__ == '__main__':
    from argparse import ArgumentParser
    from glob import glob

    parser = ArgumentParser()
    parser.add_argument("--dir_dataset", "-dd", type=str, default="/users/gyungin/datasets/ImageNet2012")
    parser.add_argument("--model_name", '-m', type=str, default="ViT-L/14@336px")
    parser.add_argument("--split", '-s', type=str, default="train")
    parser.add_argument("--batch_size", '-b', type=int, default=16)
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    dir_dataset = args.dir_dataset
    if args.size is None:
        size = {
            "RN50": 224,
            "RN50x16": 384,
            "RN50x64": 448,
            "ViT-B/32": 224,
            "ViT-B/16": 224,
            "ViT-L/14": 224,
            "ViT-L/14@336px": 336
        }[args.model_name]
    else:
        assert args.size in [224, 336, 384, 448], ValueError(args.size)
        size = args.size

    fp: str = f"{dir_dataset}/filename_to_{args.model_name.replace('/', '_').replace('-', '_').replace('@', '_')}_{args.split}_img_embedding_test.pkl"
    print(f"the img embedding will be stored at {fp}")

    p_imgs: List[str] = sorted(glob(f"{dir_dataset}/{args.split}/**/*.JPEG"))

    dataset = ImageNet1KDataset(
        dir_dataset=dir_dataset,
        split=args.split,
        size=size
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=False
    )

    extract_image_embeddings(
        p_imgs=p_imgs,
        dataloader=dataloader,
        model_name=args.model_name,
        size=size,
        fp=fp
    )