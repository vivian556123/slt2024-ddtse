python enhancement.py \
    --test_dir /Libri2Mix_mel_data/Libri2Mix/wav16k/min/test \
    --enhanced_dir /path_to_save_inferenced_samples \
    --ckpt /stage1.ckpt \
    --condition yes \
    --algorithm_type DDTSE \
    --seed 100 \