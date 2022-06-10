#!/usr/bin/env bash

if [ "$#" -eq  "0" ]
   then
     python3 ../eval.py --p_config "../configs/reco_kitti_step.yaml"
 else
     python3 ../eval.py --p_config "../configs/reco_kitti_step.yaml" --split "$1"
fi