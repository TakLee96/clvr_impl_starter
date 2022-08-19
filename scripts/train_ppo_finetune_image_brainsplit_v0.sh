PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='BrainSplitImageActorCritic' \
    --exp_name='ppo_finetune_image_brainsplit_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=0 \
    --savedir="./logs/reward_model/step-999.pth"
