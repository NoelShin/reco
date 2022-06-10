#!/usr/bin/env bash

if [ "$#" -eq  "0" ]
   then
     python3 ../eval.py --p_config "../configs/reco_cityscapes.yaml"
 else
     python3 ../eval.py --p_config "../configs/reco_cityscapes.yaml" --split "$1"
fi