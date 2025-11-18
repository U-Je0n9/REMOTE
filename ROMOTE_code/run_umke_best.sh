#!/bin/bash

DATASET_NAME="UMKE"
BERT_NAME='bert-base-multilingual-uncased'
VIT_NAME='openai/clip-vit-base-patch32'
CUDA_VISIBLE_DEVICES=0

python /home/ujeong/KETI/REMOTE/ROMOTE_code/run_umke_best.py\
        --dataset_name=${DATASET_NAME} \
        --vit_name=${VIT_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=3 \
        --batch_size=16 \
        --lr=1e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --use_dep \
        --use_box \
        --max_seq=256 \
        --save_path="ckpt"