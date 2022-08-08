PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_lr2_v0' \
    --cpu=1 --steps=4000 --epochs 250 --seed=0 --pi_lr='5e-5' --vf_lr='5e-5' \
    --savedir="./logs/reward_model/step-999.pth"
