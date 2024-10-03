echo "Compute intrinsics, undistort images and generate json files. This may take a while"
python process_data/convert_tnt_to_json.py \
        --tnt_path /your/data/path \
        --run_colmap \
        --export_json 