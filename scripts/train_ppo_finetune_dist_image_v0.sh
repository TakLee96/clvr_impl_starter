PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_dist_image_v0' \
    --cpu=4 --steps=4000 --epochs 1000 --seed=0 \
    --savedir="./logs/encoder_dist/step-499.pth"
