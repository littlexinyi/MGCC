#!/bin/bash
cd src

python train.py --model_name 'MultiGrained_clip_clip_Rv0.4_Rt0.5' \
--GPU_id 0 \
--lr 0.0001 \
--dataset 'ICFG-PEDES' \
--epoch 60 \
--dataroot '../dataset/ICFG-PEDES/' \
--class_num 3102 \
--feature_length 768 \
--train_dt_type 0 \
--test_dt_type 0 \
--image_fintune \
--text_fintune \
--txt_model 'CLIP' \
--img_model 'CLIP' \
--Topk_Selection \
--Rv 0.4 \
--Rt 0.5 \
--mode 'train' \
--batch_size 128 \
--loss_type 'InfoNCE' \
--img_text_logits \
--patch_word_logits \
--img_word_logits \
--patch_text_logits

