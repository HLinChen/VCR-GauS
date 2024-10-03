export CUDA_VISIBLE_DEVICES=0

DOMAIN_TYPE=indoor
DATADIR=/your/data/path

CODE_PATH=/your/dsine/code/path
CKPT=/your/dsine/code/path/checkpoints/dsine.pt

for SCENE in Barn Caterpillar Courthouse Ignatius Meetingroom Truck;
do
    SCENE_PATH=${DATADIR}/${SCENE}
    # dsine
    python -W ignore process_data/extract_normal.py \
            --dsine_path ${CODE_PATH} \
            --ckpt ${CKPT} \
            --img_path ${SCENE_PATH}/images \
            --intrins_path ${SCENE_PATH}/ \
            --output_path ${SCENE_PATH}/normals
done