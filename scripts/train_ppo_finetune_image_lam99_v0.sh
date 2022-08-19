PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_lam99_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=0 --lam=0.99 \
    --savedir="./logs/reward_model/step-999.pth"
