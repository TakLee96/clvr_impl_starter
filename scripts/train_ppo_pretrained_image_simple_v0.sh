PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='SimpleImageActorCritic' \
    --exp_name='ppo_pretrained_image_simple_v0' \
    --freeze --cpu=4 --steps=4000 --epochs 500 --seed=0 \
    --savedir="./logs/reward_model/step-999.pth"
