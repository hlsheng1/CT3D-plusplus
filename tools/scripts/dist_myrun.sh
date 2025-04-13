#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

python -m torch.distributed.launch --nproc_per_node=${NGPUS}  train.py --launcher pytorch --cfg_file cfgs/kitti_models/rdiou_ct3d.yaml --extra_tag 'rdiou_ct3d'
python -m torch.distributed.launch --nproc_per_node=${NGPUS}  train.py --launcher pytorch --cfg_file cfgs/kitti_models/rdiou_voxel_rcnn_car.yaml  --extra_tag 'rdiou_voxelrcnn'

