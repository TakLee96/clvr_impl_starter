PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_gamma09_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=0 --gamma=0.9 \
    --savedir="./logs/reward_model/step-999.pth"
