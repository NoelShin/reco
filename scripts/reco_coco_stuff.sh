#!/usr/bin/env bash

if [ "$#" -eq  "0" ]
   then
     python3 ../eval.py --p_config "../configs/reco_coco_stuff.yaml"
 else
     python3 ../eval.py --p_config "../configs/reco_coco_stuff.yaml" --split "$1"
fi