from typing import Dict, List, Optional, Tuple, Union
import pickle as pkl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.utils import get_model
from utils.extract_text_embeddings import prompt_engineering


class ReCo:
    def __init__(
            self,
            k: int,
            index_dataset: Optional[Dataset] = None,
            encoder_arch: str = "DeiT-S/16-SIN",
            dense_clip_arch: str = "RN50x16",
            dense_clip_inference: bool = True,
            categories: Optional[List[str]] = None,
            context_elimination: bool = True,
            context_categories: Optional[List[str]] = ["tree", "sky", "building", "road", "person"],
            text_attention: bool = True,
            device: torch.device = torch.device("cuda:0"),
            p_category_to_img_feature: Optional[str] = '',
            visualise: bool = False,
            dir_ckpt: Optional[str] = None
    ):
        self.k: int = k
        self.dense_clip_arch: str = dense_clip_arch
        self.dense_clip: callable = get_model(dense_clip_arch).to(device)
        self.dense_clip_inference: bool = dense_clip_inference
        self.encoder: callable = get_model(encoder_arch).to(device)

        self.index_dataset: Optional[Dataset] = index_dataset
        self.context_elimination: bool = context_elimination
        self.context_categories: Optional[List[str]] = context_categories
        self.text_attention: bool = text_attention
        self.device: torch.device = torch.device("cuda:0")
        self.visualise: bool = visualise
        self.dir_ckpt: Optional[str] = dir_ckpt
        if self.dir_ckpt is not None:
            os.makedirs(self.dir_ckpt, exist_ok=True)

        try:
            # load text and image features for target categories
            self.category_to_img_feature: Dict[str, torch.Tensor] = pkl.load(open(p_category_to_img_feature, "rb"))
            self.categories: List[str] = list(self.category_to_img_feature.keys())
            self.category_to_text_feature: Dict[str, torch.Tensor] = prompt_engineering(
                categories=self.categories, model_name=dense_clip_arch
            )
            print(f"p_category_to_img_feature is loaded from {p_category_to_img_feature}.")

        except FileNotFoundError:
            # compute text and image features for target categories
            assert index_dataset is not None, ValueError(type(index_dataset))
            assert os.path.exists(os.path.dirname(p_category_to_img_feature)),\
                FileNotFoundError(os.path.dirname(p_category_to_img_feature))

            # first compute text and image features for context categories
            self.category_to_context_img_feature: Dict[str, torch.Tensor] = dict()
            if context_elimination and len(self.context_categories) > 0:
                self.category_to_context_text_feature = prompt_engineering(
                    categories=self.context_categories, model_name=dense_clip_arch
                )

                for context_category in self.context_categories:
                    imgs: torch.Tensor = self._retrieve(category=context_category)
                    context_img_feature: torch.Tensor = self._cosegment(
                        imgs=imgs,
                        category=context_category,
                        text_feature=self.category_to_context_text_feature[context_category],
                        category_to_context_img_feature=self.category_to_context_img_feature
                    )
                    self.category_to_context_img_feature[context_category] = context_img_feature

            if categories is not None:
                print(f"A category-to-image-feature file will be stored at {p_category_to_img_feature}.")
                print(f"Extracting reference image features for", categories)
                self.categories: List[str] = categories
                self.category_to_text_feature: Dict[str, torch.Tensor] = prompt_engineering(
                    categories=self.categories, model_name=dense_clip_arch
                )
                self.category_to_img_feature: Dict[str, torch.Tensor] = {
                    category: self._get_category_img_feature(category=category) for category in categories
                }

                pkl.dump(self.category_to_img_feature, open(p_category_to_img_feature, "wb"))
                print(f"A category-to-image-feature file is stored at {p_category_to_img_feature}.")

            else:
                self.categories: List[str] = []
                self.category_to_text_feature: Dict[str, torch.Tensor] = {}
                self.category_to_img_feature: Dict[str, torch.Tensor] = {}

    def _retrieve(self, category: str) -> torch.Tensor:
        self.index_dataset.set_category(category=category)
        dict_data: dict = self.index_dataset[0]
        imgs: torch.Tensor = dict_data["imgs"]  # n_imgs x 3 x h x w
        return imgs.to(self.device)

    def _get_text_attention(
            self,
            imgs: torch.Tensor,  # b x 3 x H x W
            text_feature: torch.Tensor,  # c or n x c,
            output_size: Optional[Tuple] = None,  # (h, w)
            temperature: float = 10.0
    ) -> torch.Tensor:
        img_features: torch.Tensor = self.dense_clip(imgs)
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        if len(text_feature.shape) == 1:
            weights: torch.Tensor = torch.sigmoid(temperature * torch.einsum("bchw,c->bhw", img_features, text_feature))

            if output_size is not None:
                # resize the weights to the spatial size of features if necessary
                weights = F.interpolate(
                    weights[:, None], size=output_size, mode="bilinear", align_corners=False
                )[:, 0]  # n_imgs x h x w

        elif len(text_feature.shape) == 2:
            weights: torch.Tensor = torch.sigmoid(temperature * torch.einsum("bchw,nc->bnhw", img_features, text_feature))
            # n_imgs x h x w
            if output_size is not None:
                weights = F.interpolate(weights, size=output_size, mode="bilinear", align_corners=False)
        else:
            raise ValueError
        return weights

    def _get_context_weights(self, features: torch.Tensor, context_img_features: torch.Tensor) -> torch.Tensor:
        # features: n_imgs x n_dims x h x w
        # context_img_embeddings: n_context_cats x n_dims
        assert features.dim() == 4, f"{features.dim()} != 4"
        assert context_img_features.dim() == 2, f"{context_img_features.dim()} != 2"
        weights: torch.Tensor = torch.einsum(
            "bchw,nc->bnhw",
            features / features.norm(dim=1, keepdim=True),
            context_img_features / context_img_features.norm(dim=1, keepdim=True)
        )
        return torch.sigmoid(weights)

    def _get_reference_image_feature(
            self,
            features: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            return_seed_features: bool = False,
            ref_type: str = "avg"
    ) -> torch.Tensor:
        assert ref_type in ["avg", "weighted", "winner"]
        assert features.dim() == 4, ValueError(f"{features.dim()} != 4")
        n_imgs, c, h, w = features.shape

        # features_flat: n_imgs x c x h x w -> c x n_imgs x h x w -> c x (n_imgs * h * w) -> (n_imgs * h * w) x c
        features_flat = features.permute(1, 0, 2, 3).flatten(start_dim=1).t()

        # normalise features so that adj_max is [-1, 1]
        features_flat = features_flat / features_flat.norm(dim=-1, keepdim=True)

        # adj_mat: (n_imgs * h * w) x (n_imgs * h * w)
        adj_mat: torch.FloatTensor = features_flat @ features_flat.t()
        assert -1. - 1e-3 <= adj_mat.min(), ValueError(f"{adj_mat.min()} < -1")
        assert adj_mat.max() <= 1. + 1e-3, ValueError(f"{adj_mat.max()} > 1")

        if weights is not None:
            assert weights.dim() == 3, f"{weights.dim()} != 3 (n_imgs x h x w)"
            assert 0 <= weights.min(), ValueError(f"{adj_mat.min()} < 0")
            assert weights.max() <= 1, ValueError(f"{adj_mat.min()} > 1")

            weights_flat = weights.flatten()  # n_imgs * h * w
            adj_mat = adj_mat * weights_flat[None]

        # grid: (n_imgs * h * w) x n_imgs
        grid: torch.Tensor = torch.zeros((len(adj_mat), n_imgs), dtype=torch.float32, device=adj_mat.device)

        start_col: int = 0
        for num_img in range(n_imgs):
            end_col: int = start_col + h * w
            grid[:, num_img] = torch.max(adj_mat[:, start_col: start_col + end_col], dim=1).values
            start_col = end_col
        avg_grid = torch.mean(grid, dim=1)  # (n_imgs * h * w)

        seed_locs: list = list()
        seed_features: list = list()
        start_row: int = 0
        for num_img in range(n_imgs):
            end_row: int = start_row + h * w
            index_flat = torch.argmax(avg_grid[start_row: end_row]).item()
            start_row = end_row

            index_2d = [index_flat // w, index_flat % w]
            seed_locs.append(index_2d)
            seed_features.append(features[num_img, :, index_2d[0], index_2d[1]])
        seed_features: torch.Tensor = torch.stack(seed_features, dim=0)

        if ref_type == "winner":
            seed_features_norm = seed_features / (torch.linalg.norm(seed_features, ord=2, dim=-1, keepdim=True) + 1e-7)
            sim_mat: torch.Tensor = seed_features_norm @ seed_features_norm.t()  # n_imgs x n_imgs

            index_winner = torch.argmax(sim_mat.sum(dim=1))
            ref_img_feature = seed_features[index_winner]

        elif ref_type == "weighted":
            weights = torch.arange(1.0, 0., -1 / len(seed_features), device=seed_features.device)  # n_imgs
            ref_img_feature = torch.sum(seed_features * weights[..., None], dim=0) / weights.sum()

        else:
            ref_img_feature: torch.Tensor = seed_features.mean(dim=0)

        ref_img_feature = ref_img_feature / ref_img_feature.norm(dim=0, keepdim=True)

        if return_seed_features:
            return ref_img_feature, seed_features
        else:
            return ref_img_feature

    def _cosegment(
            self,
            imgs: torch.Tensor,
            category: Optional[str] = None,
            text_feature: Optional[torch.Tensor] = None,
            category_to_context_img_feature: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if category_to_context_img_feature is None:
            category_to_context_img_feature = {}

        if category in category_to_context_img_feature:
            # this can be modified s.t. categories closed enough to one another share the same image feature
            return category_to_context_img_feature[category]
        elif category == "vegetation":
            # for the Cityscapes and KITTI-STEP datasets
            return category_to_context_img_feature["tree"]

        features: torch.FloatTensor = self.encoder(imgs.to(device=self.device))  # n_imgs x c x h x w

        n_imgs, c, h, w = features.shape
        weights = torch.ones(size=(n_imgs, h, w), device=features.device)  # n_imgs x h x w

        if self.text_attention:
            assert text_feature is not None, ValueError(f"A category word should be given for text attention.")
            # n_imgs x h x w
            text_attention: torch.Tensor = self._get_text_attention(
                imgs=imgs, text_feature=text_feature, output_size=(h, w)
            )

            weights = weights * text_attention

        if self.context_elimination and len(category_to_context_img_feature) > 0:
            context_weights: torch.Tensor = self._get_context_weights(
                features=features,
                context_img_features=torch.stack(list(category_to_context_img_feature.values()), dim=0)
            )  # n_imgs x n_context_cats x h x w

            for num_context_cat in range(context_weights.shape[1]):
                weights *= (1 - context_weights[:, num_context_cat, ...])

        assert 0 - 1e-5 <= weights.min(), ValueError(f"{weights.min()}")
        assert weights.max() <= 1.0 + 1e-5, ValueError(f"{weights.max()}")

        ref_img_feature: torch.Tensor = self._get_reference_image_feature(features=features, weights=weights)  # n_dims

        if self.visualise:
            max_n_imgs: int = 5
            nrows, ncols = 1, max_n_imgs  # nrows: args.n_imgs
            scale = 3
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * scale, nrows * scale), squeeze=False)
            alpha = 0.5
            cmap = "jet"

            for i in range(nrows):
                for j in range(ncols):
                    pil_img: np.ndarray = imgs[j].cpu().numpy()  # 3 x h x w
                    H, W = pil_img.shape[-2:]
                    pil_img = pil_img * np.array([0.229, 0.224, 0.225])[:, None, None]
                    pil_img = pil_img + np.array([0.485, 0.456, 0.406])[:, None, None]
                    pil_img = pil_img * 255.0
                    pil_img = np.clip(pil_img, 0, 255)
                    pil_img: Image.Image = Image.fromarray(pil_img.astype(np.uint8).transpose(1, 2, 0))
                    ax[i, j].imshow(pil_img)  # base image

                    sim_map = torch.einsum(
                        "chw,c->hw",
                        features[j] / features[j].norm(dim=0, keepdim=True),
                        ref_img_feature
                    )
                    sim_map = F.interpolate(sim_map[None, None], size=(H, W), mode="bilinear").squeeze()  # H x W
                    ax[i, j].imshow(sim_map.cpu().numpy(), alpha=alpha, cmap=cmap)
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])

            plt.tight_layout(pad=0.5)
            plt.savefig(f"{self.dir_ckpt}/coseg_{category}.png")
            plt.close()

        return ref_img_feature

    @torch.no_grad()
    def _get_category_img_feature(self, category: str) -> torch.Tensor:
        imgs: torch.Tensor = self._retrieve(category=category)
        img_feature: torch.Tensor = self._cosegment(
            imgs=imgs,
            category=category,
            text_feature=self.category_to_text_feature[category],
            category_to_context_img_feature=self.category_to_context_img_feature
        )
        return img_feature

    def __call__(
            self,
            img: torch.Tensor,
            categories: Optional[Union[str, List[str]]] = None,
            device: torch.device = torch.device("cuda:0")
    ):
        features: torch.Tensor = self.encoder(img.to(device=self.device))  # b x c x h x w

        if categories is not None:
            if isinstance(categories, str):
                categories: List[str] = [categories]

            # extract text embeddings
            self.category_to_text_feature: Dict[str, torch.Tensor] = prompt_engineering(
                categories=categories, model_name=self.dense_clip_arch
            )

            # extract reference image embeddings
            self.category_to_img_feature: Dict[str, torch.Tensor] = {
                category: self._get_category_img_feature(category=category) for category in categories
            }

        dt: torch.Tensor = torch.einsum(
            "bchw,nc->bnhw",
            features / features.norm(dim=1, keepdim=True),
            torch.stack(list(self.category_to_img_feature.values()), dim=0)  # already normalised
        ).sigmoid()

        dt = torch.nn.functional.interpolate(
            dt.to(torch.float32),
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=False
        )  # 1 x n_dims x H x W

        if self.dense_clip_inference:
            # language-attention
            text_attention = self._get_text_attention(
                img.to(device=device),
                text_feature=torch.stack(list(self.category_to_text_feature.values()), dim=0),
                output_size=dt.shape[-2:]
            )

            dt = dt * text_attention
        return dt


if __name__ == '__main__':
    # extract image reference embeddings for categories of ImageNet1K.
    import os
    from argparse import ArgumentParser, Namespace
    import json
    import yaml
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    from networks.reco import ReCo

    parser = ArgumentParser("ReCo Evaluation")
    parser.add_argument("--p_config", type=str, default="", required=True)
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--size", type=int, default=384, help="img size of training imgs")
    args = parser.parse_args()

    args: Namespace = parser.parse_args()
    base_args = yaml.safe_load(open(f"{args.p_config}", 'r'))

    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)

    def get_experim_name(args: Namespace) -> str:
        # decide an experim name
        list_keywords: List[str] = [
            args.encoder_arch.replace('-', '_').replace('/', '_').lower(), f"in_{args.imagenet_split}"
        ]
        if args.context_elimination:
            list_keywords.append("ce")
        if args.text_attention:
            list_keywords.append("ta")
        list_keywords.append(args.suffix) if args.suffix != '' else None
        return '_'.join(list_keywords)

    experim_name = get_experim_name(args)

    # add "dc" if args.dense_clip_inference is True. Note that this does not affect image feature.
    _experim_name = experim_name + "_dc" if args.dense_clip_inference else experim_name

    dir_ckpt: str = f"{args.dir_ckpt}/{args.dataset_name}/{args.split}/reco/{_experim_name}/k{args.n_imgs:03d}"
    dir_dt_masks = f"{dir_ckpt}/dt"
    dir_dt_masks_crf = f"{dir_ckpt}/dt_crf"
    os.makedirs(dir_dt_masks, exist_ok=True)
    os.makedirs(dir_dt_masks_crf, exist_ok=True)

    print(f"\n====={dir_ckpt} is created.=====\n")
    # json.dump(vars(args), open(f"{dir_ckpt}/config.json", 'w'), indent=2, sort_keys=True)

    p_category_to_img_feature: str = f"{args.dir_dataset}/{experim_name}_cat_to_img_feature_k{args.n_imgs}.pkl"

    # load an index dataset if needed
    if not os.path.exists(p_category_to_img_feature):
        from datasets.imagenet1k import ImageNet1KDataset

        index_dataset = ImageNet1KDataset(
            dir_dataset=args.dir_imagenet,
            split=args.imagenet_split,
            k=args.n_imgs,
            model_name=args.clip_arch
        )
        print(f"ImageNet dataset ({args.imagenet_split}) is loaded.")
    else:
        index_dataset = None

    # load categories for ImageNet1k
    # copied from https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
    label_id_to_wnid_cat = json.load(open("/users/gyungin/datasets/ImageNet2012/imagenet_class_index.json", "r"))

    # exclude crane and maillot categories
    categories = list()
    wnid_to_cat: dict = dict()
    for label_id, wnid_cat in label_id_to_wnid_cat.items():
        wnid, cat = wnid_cat
        if cat in ["crane", "maillot"]:
            continue
        categories.append(cat.replace('_', ' '))  # replace an underscore by a space

    with open("imagenet1k_categories.txt", 'w') as f:
        for c in categories:
            f.write(f"{c}\n")
        f.close()

    device: torch.device = torch.device("cuda:0")

    reco = ReCo(
        index_dataset=index_dataset,
        k=args.n_imgs,
        categories=categories,
        encoder_arch=args.encoder_arch,
        dense_clip_arch=args.dense_clip_arch,
        dense_clip_inference=args.dense_clip_inference,
        p_category_to_img_feature=p_category_to_img_feature,
        text_attention=args.text_attention,
        context_elimination=args.context_elimination,
        context_categories=args.context_categories,
        device=device,
        visualise=False,
        dir_ckpt=dir_ckpt
    )