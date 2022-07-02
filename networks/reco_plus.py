from typing import Optional
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from networks.deeplab.modeling import deeplabv3plus_mobilenet, deeplabv3plus_resnet50, deeplabv3plus_resnet101
from networks.deeplab import convert_to_separable_conv, set_bn_momentum
from metrics.average_meter import AverageMeter
from utils.crf import batched_crf


class ReCoPlus:
    def __init__(
            self,
            dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            iter_train: int = 20000,
            segmentor_name: str = "deeplabv3plus_mobilenet",
            optimiser_cfg: Optional[dict] = None,
            scheduler_cfg: Optional[dict] = None,
            separable_conv: bool = True,
            ignore_index: int = 255,
            device: torch.device = torch.device("cuda:0"),
            palette: Optional[dict] = None,
            dir_ckpt: Optional[str] = None
    ):
        self.dataset: Dataset = dataset
        self.dataset_name: str = dataset.name
        self.eval_dataset: Optional[Dataset] = eval_dataset
        if eval_dataset is not None:
            assert dataset.name == eval_dataset.name,\
                ValueError(f"Train and eval datasets have different names ({dataset.name} != {eval_dataset.name})!")

        self.iter_train: int = iter_train
        self.segmentor: torch.nn.Module = self._build_segmentor(
            n_categories=dataset.n_categories, segmentor_name=segmentor_name, separable_conv=separable_conv
        ).to(device)

        self.criterion: nn.Module = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.optimiser: torch.optim.Optimizer = self._build_optimiser(
            segmentor=self.segmentor, optimiser_cfg=optimiser_cfg
        )
        self.lr_scheduler = self._build_lr_scheduler(optimiser=self.optimiser, scheduler_cfg=scheduler_cfg)
        self.device: torch.device = device
        self.n_categories: int = dataset.n_categories
        self.palette: dict = palette
        self.dir_ckpt: str = dir_ckpt
        print(f"The number of categories: {self.n_categories}")
        self.best_miou: float = 0.

        # create a checkpoint directory
        if self.dir_ckpt is not None:
            os.makedirs(self.dir_ckpt, exist_ok=True)
            print(f"A directory is created ({self.dir_ckpt}).")

    @staticmethod
    def _build_segmentor(
            n_categories: int, segmentor_name: str = "deeplabv3plus_mobilenet", separable_conv: bool = True
    ) -> torch.nn.Module:
        if segmentor_name == "deeplabv3plus_mobilenet":
            segmentor = deeplabv3plus_mobilenet(num_classes=n_categories)
        elif segmentor_name == "deeplabv3plus_resnet50":
            segmentor = deeplabv3plus_resnet50(num_classes=n_categories)
        elif segmentor_name == "deeplabv3plus_resnet101":
            segmentor = deeplabv3plus_resnet101(num_classes=n_categories)
        else:
            raise ValueError(f"Invalid segmentor: {segmentor_name}")

        if separable_conv and 'plus' in segmentor_name:
            convert_to_separable_conv(segmentor.classifier)
        set_bn_momentum(segmentor.backbone, momentum=0.01)
        print(f"{segmentor_name} is loaded.")
        return segmentor

    def _build_optimiser(
            self, segmentor: torch.nn.Module, optimiser_cfg: Optional[dict] = None
    ) -> torch.optim.Optimizer:
        if optimiser_cfg is None:
            lr: float = 5e-4
            weight_decay: float = 2e-4
            betas = (0.9, 0.999)
            optimiser = torch.optim.Adam(
                params=[
                    {'params': segmentor.backbone.parameters(), 'lr': 0.1 * lr},
                    {'params': segmentor.classifier.parameters(), 'lr': lr},
                ], lr=lr, weight_decay=weight_decay, betas=betas
            )
        else:
            raise NotImplementedError

        return optimiser

    def _build_lr_scheduler(
            self, optimiser: torch.optim.Optimizer, scheduler_cfg: Optional[dict] = None
    ):
        if scheduler_cfg is None:
            total_iters: int = self.iter_train
            from utils.scheduler import PolyLR
            lr_scheduler = PolyLR(optimiser, total_iters, power=0.9)
        else:
            raise NotImplementedError

        return lr_scheduler

    def __call__(
            self,
            iter_log: int = 100,
            iter_val: int = 1000,
            batch_size: int = 8,
            n_workers: int = 8
    ) -> None:
        self.dataset.augmentation_on()
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True, shuffle=True
        )
        iter_dataloader = iter(dataloader)
        pbar = tqdm(range(1, self.iter_train + 1))  # iter starts from 1
        running_score = RunningScore(self.n_categories)
        loss_meter = AverageMeter()
        for num_iter in pbar:
            try:
                dict_data = next(iter_dataloader)

            except StopIteration:
                iter_dataloader = iter(dataloader)
                dict_data = next(iter_dataloader)
                running_score.reset()
                loss_meter.reset()

            # forward
            img: torch.Tensor = dict_data["img"].to(self.device)
            gt: torch.Tensor = dict_data["pseudo_gt"].to(self.device)
            dt = self.segmentor(img)

            # backward
            loss = self.criterion(dt, gt)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.lr_scheduler.step()

            # compute metrics
            dt_argmax = torch.argmax(dt, dim=1)
            running_score.update(gt.cpu().numpy(), dt_argmax.detach().cpu().numpy())
            loss_meter.update(loss.detach().cpu().item(), 1)

            miou = running_score.get_scores()[0]["Mean IoU"]
            pixel_acc = running_score.get_scores()[0]["Pixel Acc"]
            pbar.set_description(
                f"({num_iter}/{self.iter_train}) | "
                f"Loss: {loss_meter.avg:.3f} | "
                f"mIoU: {miou:.3f} | "
                f"pixel acc.: {pixel_acc:.3f}"
            )

            # save training metrics
            if num_iter % iter_log == 0 and self.dir_ckpt is not None:
                results: dict = {"num_iter": num_iter}
                results.update(running_score.get_scores()[0])
                results.update(running_score.get_scores()[1])

                if num_iter == iter_log:
                    json.dump(results, open(f"{self.dir_ckpt}/training_metrics.json", 'w'))
                else:
                    with open(f"{self.dir_ckpt}/training_metrics.json", 'a') as f:
                        f.write('\n')
                        json.dump(results, f)
                        f.close()

            # evaluate the model
            if num_iter % iter_val == 0:
                self.validate(num_iter=num_iter)

    def visualise(
            self,
            fp: str,
            img: np.ndarray,
            gt: np.ndarray,
            dt: np.ndarray,
            palette: dict,
            dt_crf: Optional[np.ndarray] = None,
    ):
        def colourise_label(label: np.ndarray, palette: dict, ignore_index: int = -1) -> np.ndarray:
            h, w = label.shape[-2:]
            coloured_label = np.zeros((h, w, 3), dtype=np.uint8)

            unique_label_ids = np.unique(label)
            for label_id in unique_label_ids:
                if label_id == ignore_index:
                    continue
                coloured_label[label == label_id] = palette[label_id]
            return coloured_label

        img = img * np.array([0.229, 0.224, 0.225])[:, None, None]
        img = img + np.array([0.485, 0.456, 0.406])[:, None, None]
        img = img * 255.0
        img = np.clip(img, 0, 255)
        img: Image.Image = Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0))

        coloured_gt: np.ndarray = colourise_label(label=gt, palette=palette)  # h x w x 3
        coloured_dt: np.ndarray = colourise_label(label=dt, palette=palette)  # h x w x 3
        if dt_crf is not None:
            coloured_dt_crf: np.ndarray = colourise_label(label=dt_crf, palette=palette)  # h x w x 3

        ncols = 4 if dt_crf is not None else 3
        fig, ax = plt.subplots(nrows=1, ncols=ncols, squeeze=False, figsize=(ncols * 3, 3))
        for i in range(1):
            for j in range(ncols):
                if j == 0:
                    ax[i, j].imshow(img)
                elif j == 1:
                    ax[i, j].imshow(coloured_gt)
                elif j == 2:
                    ax[i, j].imshow(coloured_dt)
                elif j == 3 and dt_crf is not None:
                    ax[i, j].imshow(coloured_dt_crf)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        plt.tight_layout(pad=0.5)
        plt.savefig(fp)
        plt.close()

    @torch.no_grad()
    def validate(self, num_iter: int) -> None:
        print(f"Evaluating at iter {num_iter} (/{self.iter_train})...")
        os.makedirs(f"{self.dir_ckpt}/{num_iter:05d}".replace('-000', '-'), exist_ok=True)

        self.segmentor.eval()

        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=1 if self.dataset_name == "kitti_step" else 16,
            num_workers=16,
            pin_memory=True
        )
        iter_dataloader, pbar = iter(dataloader), tqdm(range(len(dataloader)))
        running_score = RunningScore(self.n_categories)
        running_score_crf = RunningScore(self.n_categories)
        with Pool(16) as pool:
            for num_batch in pbar:
                dict_data = next(iter_dataloader)

                img: torch.Tensor = dict_data["img"].to(self.device)  # b x 3 x H x W
                gt: np.ndarrray = dict_data["gt"].numpy()  # b x H x W

                dt: torch.Tensor = self.segmentor(img)  # b x n_cats x H x W

                dt_argmax: torch.Tensor = torch.argmax(dt, dim=1)  # b x H x W
                dt_argmax: np.ndarray = dt_argmax.cpu().numpy()  # b x H x W
                running_score.update(gt, dt_argmax)

                dt_crf_argmax = batched_crf(pool, img, torch.log_softmax(dt, dim=1)).argmax(1).to(self.device)  # b x H x W
                dt_crf_argmax: np.ndarray = dt_crf_argmax.cpu().numpy()  # b x H x W
                running_score_crf.update(gt, dt_crf_argmax)

                miou = running_score.get_scores()[0]["Mean IoU"]
                pixel_acc = running_score.get_scores()[0]["Pixel Acc"]

                miou_crf = running_score_crf.get_scores()[0]["Mean IoU"]
                pixel_acc_crf = running_score_crf.get_scores()[0]["Pixel Acc"]

                pbar.set_description(
                    f"mIoU: {miou:.3f} ({miou_crf:.3f}) | "
                    f"pixel acc.: {pixel_acc:.3f} ({pixel_acc_crf:.3f})"
                )

                if num_batch % 50 == 0 and self.dir_ckpt is not None and self.palette is not None:
                    self.visualise(
                        fp=f"{self.dir_ckpt}/{num_iter:05d}/{num_batch:04d}.png".replace('-000', '-'),
                        img=img[0].cpu().numpy(),
                        gt=gt[0],
                        dt=dt_argmax[0],
                        dt_crf=dt_crf_argmax[0],
                        palette=self.palette
                    )
        print(
            f"Validation: ({num_iter}/{self.iter_train}) | "
            f"mIoU: {miou:.3f} ({miou_crf:.3f}) | "
            f"pixel acc.: {pixel_acc:.3f} ({pixel_acc_crf:.3f})"
        )

        # save results
        if self.dir_ckpt is not None:
            results = running_score.get_scores()[0]
            results.update(running_score.get_scores()[1])
            json.dump(results, open(f"{self.dir_ckpt}/{num_iter:05d}/results.json".replace('-000', '-'), "w"))

            results_crf = running_score_crf.get_scores()[0]
            results_crf.update(running_score_crf.get_scores()[1])
            json.dump(results_crf, open(f"{self.dir_ckpt}/{num_iter:05d}/results_crf.json".replace('-000', '-'), "w"))

        # save model weights
        if miou_crf > self.best_miou and num_iter != -1:
            self.best_miou = miou_crf
            torch.save(self.segmentor.state_dict(), f"{self.dir_ckpt}/best_model.pt")
            print(f"best model with an mIoU of {miou_crf:.3f} is saved.")

        self.segmentor.train()


if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    from utils.utils import get_dataset
    import os
    from typing import Optional
    import yaml
    import ujson as json
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    from PIL import Image
    from tqdm import tqdm
    from metrics.running_score import RunningScore

    parser = ArgumentParser()
    parser.add_argument("--p_config", type=str, default="", required=True)
    parser.add_argument("--p_state_dict", type=str, default=None)
    args = parser.parse_args()

    args: Namespace = parser.parse_args()
    base_args = yaml.safe_load(open(f"{args.p_config}", 'r'))

    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)

    # load a training dataset
    dataset, categories, palette = get_dataset(
        dir_dataset=args.dir_dataset,
        dataset_name=args.dataset_name,
        split="train",
        dir_pseudo_masks=args.dir_pseudo_masks
    )

    # load a validation dataset
    eval_dataset, _, _ = get_dataset(
        dir_dataset=args.dir_dataset,
        dataset_name=args.dataset_name,
        split="val"
    )

    dir_ckpt: str = f"{args.dir_ckpt}/{args.dataset_name}/val/reco_plus/{args.segmentor_name}"

    reco_plus = ReCoPlus(
        dataset=dataset,
        segmentor_name=args.segmentor_name,
        eval_dataset=eval_dataset,
        palette=palette,
        dir_ckpt=dir_ckpt
    )

    if args.p_state_dict is not None:
        state_dict = torch.load(args.p_state_dict)
        reco_plus.segmentor.load_state_dict(state_dict, strict=True)
        print(f"Pre-trained weights are loaded from {args.p_state_dict}.")
        reco_plus.validate(num_iter=-1)

    else:
        reco_plus(
            batch_size=args.batch_size,
            iter_val=args.iter_val
        )