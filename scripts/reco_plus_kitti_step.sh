#!/usr/bin/env bash

if [ "$#" -eq  "0" ]
   then
     python3 ../networks/reco_plus.py \
     --p_config "../configs/reco_plus_kitti_step.yaml"
 else
     python3 ../networks/reco_plus.py \
     --p_config "../configs/reco_plus_kitti_step.yaml" \
     --p_state_dict "$1"
     # --p_state_dict "/users/gyungin/denseclip/analysis/reco/scripts/ckpt_recoplus/kitti_step/deeplabv3plus_resnet101/best_model.pt"
fi
