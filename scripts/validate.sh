set -x

CONFIG=$1
CKPT=$2
PORT=${3:-23456}

HOST=$(hostname -i)

python ./scripts/validate.py \
    --cfg ${CONFIG} \
    --valid-batch 64 \
    --flip-test \
    --checkpoint ${CKPT} \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
