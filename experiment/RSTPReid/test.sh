#!/bin/bash
cd src

python test.py --model_name 'MultiGrained_clip_clip_Rv0.5_Rt0.5' \
--GPU_id 0 \
--lr 0.0001 \
--dataset 'RSTPReid' \
--epoch 60 \
--dataroot '../dataset/RSTPReid/' \
--class_num 3701 \
--feature_length 768 \
--test_dt_type 0 \
--image_fintune \
--text_fintune \
--txt_model 'CLIP' \
--img_model 'CLIP' \
--mode 'test' \
--Topk_Selection \
--Rv 0.5 \
--Rt 0.5 \
--batch_size 128 \
--loss_type 'InfoNCE' \
--img_text_logits \
--img_word_logits \
--patch_word_logits \
--patch_text_logits