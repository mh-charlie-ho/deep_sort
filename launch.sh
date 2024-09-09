#!/usr/bin/env bash

test_root_dir=/media/charlie/DataSet/deep_sort
sequence_dir=$test_root_dir/MOT16/test/MOT16-06
detection_dir=$test_root_dir/resources/detections/MOT16_POI_test/MOT16-06.npy
echo $sequence_dir
echo $detection_dir

# sequence_dir   | 用來追蹤的來源
# detection_dir  |
# min_confidence | 忽略比這個值還小的檢測結果，用來篩選detection檔案中的框
# nn_budget      | 對於"appearance descriptors"閾值
# display        | 顯示追蹤結果
python deep_sort_app.py --sequence_dir=$sequence_dir \
    --detection_file=$detection_dir \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True


