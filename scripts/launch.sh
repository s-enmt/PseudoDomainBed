#!/bin.sh $6

# <Arguments>
# $1 : mode 
# $2 : backbone name (resnet50, ViT-B16, etc)
# $3 : n_traials_from 
# $4 : n_trials 
# $5 : command launcher (local/multi_gpu) 
# 
# <Example>
# sh scripts/launch.sh train resnet50 10 3 local

if [ $1 = "train" ]; then
    echo "pretraining the model"
    sh scripts/train.sh $2 PACS $3 $4 $5 $6
    sh scripts/train.sh $2 VLCS $3 $4 $5 $6 
    sh scripts/train.sh $2 OfficeHome $3 $4 $5 $6
    sh scripts/train.sh $2 TerraIncognita $3 $4 $5 $6
else
    echo "Invalid option"
fi
