PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v1' \
    --model='ImageActorCritic' \
    --exp_name='ppo_pretrained_image_v1' \
    --freeze --cpu=1 --steps=1000 --epochs 200 \
    --savedir="./logs/reward_model/step-999.pth"
