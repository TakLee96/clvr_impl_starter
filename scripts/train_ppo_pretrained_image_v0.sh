PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_pretrained_image_v0' \
    --freeze --cpu=2 --steps=2000 \
    --savedir="./logs/reward_model/step-999.pth"
