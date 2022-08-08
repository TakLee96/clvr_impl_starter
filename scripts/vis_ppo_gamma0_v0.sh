PYTHONPATH=$(pwd) python spinup/plot.py \
    ./logs/ppo_state_gamma0_v0/ \
    ./logs/ppo_pretrained_image_gamma0_v0/ \
    ./logs/ppo_finetune_image_gamma0_v0/ \
    ./logs/ppo_scratch_image_gamma0_v0/
