#!/bin/bash
cd src

python test.py --model_name 'MultiGrained_clip_clip_Rv0.3_Rt0.4' \
--GPU_id 0 \
--lr 0.0001 \
--dataset 'CUHK-PEDES' \
--epoch 60 \
--dataroot '../dataset/cuhk/CUHK-PEDES/' \
--class_num 11000 \
--feature_length 768 \
--test_dt_type 0 \
--image_fintune \
--text_fintune \
--txt_model 'CLIP' \
--img_model 'CLIP' \
--Topk_Selection \
--Rv 0.3 \
--Rt 0.4 \
--mode 'test' \
--batch_size 128 \
--loss_type 'InfoNCE' \
--img_text_logits \
--img_word_logits \
--patch_word_logits \
--patch_text_logits