export CUDA_VISIBLE_DEVICES=0

DATADIR=/your/data/path
GSAM_PATH=~/code/gsam
CKPT_PATH=${GSAM_PATH}

for SCENE in Barn Caterpillar Courthouse Ignatius Meetingroom Truck;
do
    SCENE_PATH=${DATADIR}/${SCENE}
    # meething room scene_tye: indoor, others: outdoor
        if [ ${SCENE} = "Meetingroom" ]; then
            SCENE_TYPE="indoor"
        else
            SCENE_TYPE="outdoor"
        fi
    python -W ignore process_data/extract_mask.py \
            --config ${GSAM_PATH}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
            --grounded_checkpoint ${CKPT_PATH}/groundingdino_swint_ogc.pth \
            --sam_hq_checkpoint ${CKPT_PATH}/sam_hq_vit_h.pth \
            --gsam_path ${GSAM_PATH} \
            --use_sam_hq \
            --input_image ${SCENE_PATH}/images/ \
            --output_dir ${SCENE_PATH}/masks \
            --box_threshold 0.5 \
            --text_threshold 0.2 \
            --scene ${SCENE} \
            --scene_type ${SCENE_TYPE} \
            --device "cuda"
done
