PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_lr0_v0' \
    --cpu=1 --steps=4000 --epochs 250 --seed=0 --pi_lr='1e-4' \
    --savedir="./logs/reward_model/step-999.pth"
