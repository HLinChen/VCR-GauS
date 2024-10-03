GPU=0
export CUDA_VISIBLE_DEVICES=${GPU}
ls


DATASET=tnt
SCENE=Barn
NAME=${SCENE}

PROJECT=vcr_gaus

TRIAL_NAME=vcr_gaus

CFG=configs/${DATASET}/${SCENE}.yaml

DIR=/your/log/path/${PROJECT}/${DATASET}/${NAME}/${TRIAL_NAME}

python train.py \
    --config=${CFG} \
    --port=-1 \
    --logdir=${DIR} \
    --model.source_path=/your/data/path/${DATASET}/${SCENE}/ \
    --model.resolution=1 \
    --model.data_device=cpu \
    --wandb \
    --wandb_name ${PROJECT}
