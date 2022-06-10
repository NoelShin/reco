from argparse import Namespace
from typing import List


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


# evaluation script for ReCo
if __name__ == '__main__':
    import os
    from multiprocessing import Pool
    from argparse import ArgumentParser
    import json
    import yaml
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from PIL import Image
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from networks.reco import ReCo
    from metrics.running_score import RunningScore
    from datasets.coco_stuff import coco_stuff_171_to_27
    from utils.utils import get_dataset
    from utils.crf import batched_crf

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

    experim_name = get_experim_name(args)

    # add "dc" if args.dense_clip_inference is True. Note that this does not affect image feature.
    _experim_name = experim_name + "_dc" if args.dense_clip_inference else experim_name

    dir_ckpt: str = f"{args.dir_ckpt}/{args.dataset_name}/{args.split}/reco/{_experim_name}/k{args.n_imgs:03d}"
    dir_dt_masks = f"{dir_ckpt}/dt"
    dir_dt_masks_crf = f"{dir_ckpt}/dt_crf"
    os.makedirs(dir_dt_masks, exist_ok=True)
    os.makedirs(dir_dt_masks_crf, exist_ok=True)

    print(f"\n====={dir_ckpt} is created.=====\n")
    json.dump(vars(args), open(f"{dir_ckpt}/config.json", 'w'), indent=2, sort_keys=True)

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

    # load a benchmark dataset
    dataset, categories, palette = get_dataset(
        dir_dataset=args.dir_dataset,
        dataset_name=args.dataset_name,
        split=args.split,
        dense_clip_arch=args.dense_clip_arch
    )

    running_score = RunningScore(n_classes=dataset.n_categories)
    running_score_crf = RunningScore(n_classes=dataset.n_categories)

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
        visualise=True,
        dir_ckpt=dir_ckpt
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1 if args.dataset_name == "kitti_step" else args.batch_size,
        num_workers=args.n_workers,
        pin_memory=True
    )

    iter_dataloader, pbar = iter(dataloader), tqdm(range(len(dataloader)))
    list_dt: List[Image.Image] = list()
    list_dt_crf: List[Image.Image] = list()
    list_gt_filenames: List[str] = list()
    with Pool(args.n_workers + 5) as pool:
        for num_batch in pbar:
            dict_data = next(iter_dataloader)

            val_img: torch.Tensor = dict_data["img"]  # b x 3 x H x W
            val_gt: torch.LongTensor = dict_data["gt"]  # b x H x W

            dt: torch.Tensor = reco(val_img)  # b x n_cats x H x W
            dt_argmax: torch.Tensor = torch.argmax(dt, dim=1)  # b x H x W

            dt_crf_argmax = batched_crf(pool, val_img, torch.log_softmax(dt, dim=1)).argmax(1).to(device)

            if args.dataset_name == "coco_stuff":  # and not args.coarse_labels:
                dt_coarse: torch.Tensor = torch.zeros_like(dt_argmax)
                dt_coarse_crf: torch.Tensor = torch.zeros_like(dt_crf_argmax)

                for fine, coarse in coco_stuff_171_to_27.items():
                    dt_coarse[dt_argmax == fine] = coarse
                    dt_coarse_crf[dt_crf_argmax == fine] = coarse
                dt_argmax = dt_coarse
                dt_crf_argmax = dt_coarse_crf

            dt_argmax: np.ndarray = dt_argmax.cpu().numpy()  # b x H x W
            dt_crf_argmax: np.ndarray = dt_crf_argmax.cpu().numpy()  # b x H x W

            running_score.update(label_trues=val_gt.cpu().numpy(), label_preds=dt_argmax)
            running_score_crf.update(label_trues=val_gt.cpu().numpy(), label_preds=dt_crf_argmax)

            miou_crf = running_score_crf.get_scores()[0]["Mean IoU"]
            acc_crf = running_score_crf.get_scores()[0]["Pixel Acc"]

            miou = running_score.get_scores()[0]["Mean IoU"]
            acc = running_score.get_scores()[0]["Pixel Acc"]

            pbar.set_description(
                f"{experim_name} | "
                f"knn imgs: {args.n_imgs} | "
                f"mIoU (bi) {miou:.3f} ({miou_crf:.3f}) | "
                f"Pixel acc (bi) {acc:.3f} ({acc_crf:.3f})"
            )

            if num_batch <= 10:
                pil_img = val_img[0].cpu().numpy()
                pil_img = pil_img * np.array([0.229, 0.224, 0.225])[:, None, None]
                pil_img = pil_img + np.array([0.485, 0.456, 0.406])[:, None, None]
                pil_img = pil_img * 255.0
                pil_img = np.clip(pil_img, 0, 255)
                val_pil_img: Image.Image = Image.fromarray(pil_img.astype(np.uint8).transpose(1, 2, 0))

                val_gt: np.ndarray = val_gt[0].clone().squeeze(dim=0).cpu().numpy()
                h, w = dt_argmax.shape[-2:]
                unique_labels_dt = np.unique(dt_argmax[0])
                unique_labels_dt_crf = np.unique(dt_crf_argmax[0])
                unique_labels_gt = np.unique(val_gt)

                coloured_dt = np.zeros((h, w, 3), dtype=np.uint8)
                coloured_dt_bi = np.zeros((h, w, 3), dtype=np.uint8)
                coloured_gt = np.zeros((h, w, 3), dtype=np.uint8)
                for ul in unique_labels_dt:
                    if ul == -1:
                        continue
                    coloured_dt[dt_argmax[0] == ul] = palette[ul]

                for ul in unique_labels_dt_crf:
                    if ul == -1:
                        continue
                    coloured_dt_bi[dt_crf_argmax[0] == ul] = palette[ul]

                for ul in unique_labels_gt:
                    if ul == -1:
                        continue
                    coloured_gt[val_gt == ul] = palette[ul]

                nrows, ncols = 1, 4
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols * 3, nrows * 3))
                for i in range(nrows):
                    for j in range(ncols):
                        if j == 0:
                            ax[i, j].imshow(val_pil_img)
                        elif j == 1:
                            ax[i, j].imshow(coloured_gt)
                        elif j == 2:
                            ax[i, j].imshow(coloured_dt)
                        elif j == 3:
                            ax[i, j].imshow(coloured_dt_bi)
                        ax[i, j].set_xticks([])
                        ax[i, j].set_yticks([])
                plt.tight_layout(pad=0.5)
                plt.savefig(f"{dir_ckpt}/{num_batch:04d}.png")
                plt.close()

            if args.dataset_name in ["cityscapes", "coco_stuff", "kitti_step"] and args.split == "train":
                # save predictions for pseudo-label training
                for i in range(len(val_img)):
                    if args.dataset_name == "kitti_step":
                        video_id = dict_data['p_gt'][i].split('/')[-2]
                        os.makedirs(f"{dir_dt_masks}/{video_id}", exist_ok=True)
                        os.makedirs(f"{dir_dt_masks_crf}/{video_id}", exist_ok=True)
                        filename = f"{dict_data['p_gt'][i].split('/')[-1].replace('.mat', '.png')}"

                        Image.fromarray(dt_argmax[i].astype(np.uint8)).save(f"{dir_dt_masks}/{video_id}/{filename}")
                        Image.fromarray(dt_crf_argmax[i].astype(np.uint8)).save(
                            f"{dir_dt_masks_crf}/{video_id}/{filename}")

                    else:
                        filename = f"{dict_data['p_gt'][i].split('/')[-1].replace('.mat', '.png')}"

                        Image.fromarray(dt_argmax[i].astype(np.uint8)).save(f"{dir_dt_masks}/{filename}")
                        Image.fromarray(dt_crf_argmax[i].astype(np.uint8)).save(f"{dir_dt_masks_crf}/{filename}")

    results = running_score.get_scores()[0]
    results.update(running_score.get_scores()[1])

    results_crf = running_score_crf.get_scores()[0]
    results_crf.update(running_score_crf.get_scores()[1])

    json.dump(results, open(f"{dir_ckpt}/results.json", "w"))
    json.dump(results_crf, open(f"{dir_ckpt}/results_crf.json", "w"))