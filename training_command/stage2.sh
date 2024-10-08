TORCH_DISTRIBUTED_DEBUG=DETAIL
python train.py --base_dir  /LibriMix/Libri2Mix_mel_data/Libri2Mix/wav16k/min  \
 --backbone=conditionalncsnpp \
 --no_wandb \
 --condition_on_spkemb=yes \
 --condition=yes \
 --batch_size=3 \
 --gpus=8 \
 --num_frames=512 \
 --args.algorithm_type DDTSE \
 --resblock_type conditional_film_biggan\
 --middle_concat_attention=False \
 --loss_type mse \
 --return_interference \
 --sisdr -1.0 \
 --format default_gt_enroll \
 --triplet -1.0 \
 --train_noisy_data mix_both \
 --use_2_channel \
 --train_subset train-360 \
 --lr 5e-5 \
 --teacher_forcing \
 --max_threshold 0.5 \
 --ddtse_save_dir /path_ddtse_stage2 \
 --pretrained_model /path_ddtse_stage1/stage1.ckpt