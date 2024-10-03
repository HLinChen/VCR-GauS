export CUDA_VISIBLE_DEVICES=0

# DOMAIN_TYPE=outdoor
# DOMAIN_TYPE=indoor
DOMAIN_TYPE=object
DATADIR=/your/data/path/DTU_mask

CODE_PATH=/your/geowizard/path


for SCENE in scan106  scan114  scan122  scan37  scan55  scan65  scan83 scan105    scan110  scan118  scan24   scan40  scan63  scan69  scan97;
do
    SCENE_PATH=${DATADIR}/${SCENE}
    python process_data/extract_normal_geo.py \
        --code_path ${CODE_PATH} \
        --input_dir ${SCENE_PATH}/images/ \
        --output_dir ${SCENE_PATH}/ \
        --ensemble_size 3 \
        --denoise_steps 10 \
        --seed 0 \
        --domain ${DOMAIN_TYPE}
done