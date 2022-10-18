## ReCo: Retrieve and Co-segment for Zero-shot Transfer
Official PyTorch implementation for ReCo (NeurIPS 2022). Details can be found in the paper.
[[Paper]](https://arxiv.org/pdf/2206.07045.pdf) [[Project page]](https://www.robots.ox.ac.uk/~vgg/research/reco)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reco-retrieve-and-co-segment-for-zero-shot-1/unsupervised-semantic-segmentation-with-3)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-3?p=reco-retrieve-and-co-segment-for-zero-shot-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reco-retrieve-and-co-segment-for-zero-shot-1/unsupervised-semantic-segmentation-with-1)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-1?p=reco-retrieve-and-co-segment-for-zero-shot-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reco-retrieve-and-co-segment-for-zero-shot-1/unsupervised-semantic-segmentation-with-2)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-with-2?p=reco-retrieve-and-co-segment-for-zero-shot-1)


![Alt Text](project_page/resources/reco_no_loop.gif)

### Contents
* [Preparation](#preparation)
* [ReCo inference](#reco-inference)
* [ReCo+ training/inference](#reco+-training/inference)
* [Pre-trained weights](#pre-trained-weights)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

[comment]: <> (### Demo)

[comment]: <> (To be updated.)

[comment]: <> (Please visit [LINK] for the ReCo demo.)

### Preparation
#### 1. Download datasets
To evaluate ReCo, you first need to download some datasets.
Please visit following links to download datasets:
* [Cityscapes](https://www.cityscapes-dataset.com/login)
* [COCO-Stuff](https://github.com/mhamilton723/STEGO#install) (Download through STEGO codebase)
* [ImageNet2012](https://image-net.org/download.php)
* [KITTI-STEP](http://www.cvlibs.net/datasets/kitti/eval_step.php)

Note that Cityscapes, ImageNet2012, and KITTI-STEP require you to sign up an account.

To reimplement ReCo+ on COCO-Stuff as in our paper, you additionally need to download [COCO-Stuff10K](https://github.com/nightrome/cocostuff10k).

Please don't change the (sub)directory name(s) as the code assumes the original directory names.
We advise you to put the downloaded dataset(s) into the following directory structure for ease of implementation:
```bash
{your_dataset_directory}
├──cityscapes
│  ├──gtFine
│  ├──leftImg8bit
├──cocostuff
│  ├──annotations
│  ├──curated
│  ├──images
├──cocostuff10k
│  ├──annotations
│  ├──imageLists
│  ├──images
├──ImageNet2012
│  ├──train
│  ├──val
├──kitti_step
   ├──panoptic_maps
   ├──train
   ├──val
```

#### 2. Download required python packages:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
conda install -c conda-forge timm
conda install -c conda-forge opencv
conda install -c anaconda ujson
conda install -c conda-forge pyyaml
pip install opencv-python
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
pip install git+https://github.com/openai/CLIP.git
```

Additionally, please install `mmcv` following the instructions in [the official website](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

### ReCo inference
To evalute ReCo, you need to set up some directory/file paths (e.g., dataset directory). For this please
open `reco_$DATASET_NAME.yaml` file in configs directory and find "dir_ckpt" and "dir_dataset" arguments where `$DATASET_NAME` is either `cityscapes`, `coco_stuff`, or `kitti_step`.
Then, type your corresponding paths:

```yaml
dir_ckpt: [YOUR_DESIRED_CHECKPOINT_DIR]
dir_dataset: [YOUR_DATASET_DIR]
dir_imagenet: [YOUR_ImageNet2012_DIR]
```

To validate on Cityscapes, COCO-Stuff, or KITTI-STEP, move to `scripts` directory and run
```shell
bash reco_$DATASET_NAME.sh
```
Note that this will first extract and save image embeddings for the ImageNet2012 images.
This process occurs only for the first time and takes up to a few hours.
If you want to avoid this, please download the pre-computed image embeddings via this [link (~4.3 GB)](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/filename_to_ViT_L_14_336px_train_img_embedding.pkl) and put the downloaded file into your ImageNet2012 directory.

In addition, if you also want to avoid computing reference image embeddings for categories in a benchmark, please download the pre-computed reference image embeddings file for the benchmark and put it into the benchmark directory (e.g., put reference image embeddings for the Cityscapes categories into your Cityscapes directory):
* [Cityscapes](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/cityscapes/deit_s_16_sin_in_train_ce_ta_cat_to_img_feature_k50.pkl) (27 categories)
* [COCO-Stuff](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/coco_stuff/deit_s_16_sin_in_train_ce_ta_cat_to_img_feature_k50.pkl) (171 categories)
* [KITTI-STEP](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/kitti_step/deit_s_16_sin_in_train_ce_ta_cat_to_img_feature_k50.pkl) (19 categories)

### ReCo+ training/inference
Unlike ReCo, which does not involve any training, ReCo+ is trained on the training split of each benchmark.
To avoid using human-annotations, ReCo+ utilises predictions made by ReCo as pseudo-labels.

#### 1. Generate pseudo-masks
To compute pseudo-masks for training ReCo+ on Cityscapes, COCO-Stuff, or KITTI-STEP, run
```shell
bash reco_$DATASET_NAME.sh "train"
```

By default, the pseudo-masks will be stored in the dataset directory.
If you want to skip this process, please download the pre-computed pseudo-masks:
* [Cityscapes](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/cityscapes/reco_cityscapes_pseudo_masks.tar) (~14.2 MB)
* [COCO-Stuff](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/coco_stuff/reco_coco_stuff_pseudo_masks.tar) (~47 MB)
* [KITTI-STEP](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/kitti_step/reco_kitti_step_pseudo_masks.tar) (~86.3 MB)

#### 2. Training
Once pseudo-masks are created (or downloaded and uncompressed), set a path to the directory that contains the pseudo-masks in a configuration file.
For example, open the `reco_plus_cityscapes.yaml` file and change `dir_pseudo_masks` argument as appropriate.
Then, run
```shell
bash reco_plus_$DATASET_NAME.sh
```

Note that an evaluation will be made at every 1,000 iterations during training and the weights for the best model will be saved at your checkpoint directory.

#### 3. Inference
To run an inference script with pre-trained weights, please run
```shell
bash reco_plus_$DATASET_NAME.sh $PATH_TO_WEIGHTS
```

### Pre-trained weights
We provide the pre-trained weights for ReCo+:

benchmark|IoU (%)|pixel accuracy (%)|link|
:---:|:---:|:---:|:---:|
Cityscapes| 24.2 | 83.7 |[weights](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/cityscapes/reco_plus_dlp_rn101_cityscapes.pt) (~183.1 MB)
COCO-Stuff| 32.6 | 54.1 |[weights](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/coco_stuff/reco_plus_dlp_rn101_coco_stuff.pt) (~183.1 MB)
KITTI-STEP| 31.9 | 75.3 |[weights](https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/kitti_step/reco_plus_dlp_rn101_kitti_step.pt) (~183.1 MB)

### Citation
```
@article{shin2022reco,
  author = {Shin, Gyungin and Xie, Weidi and Albanie, Samuel},
  title = {ReCo: Retrieve and Co-segment for Zero-shot Transfer},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2022}
}
```

### Acknowledgements
We borrowed the code for CLIP, DeepLabv3+, DenseCLIP, DINO, ViT from
* [CLIP](https://github.com/rwightman/pytorch-image-models)
* [DeepLabv3+](https://github.com/VainF/DeepLabV3Plus-Pytorch)
* [DenseCLIP](https://github.com/chongzhou96/DenseCLIP)
* [DINO](https://github.com/facebookresearch/dino)
* [ViT](https://github.com/rwightman/pytorch-image-models)

If you have any questions about our code/implementation, please contact us at gyungin [at] robots [dot] ox [dot] ac [dot] uk.
