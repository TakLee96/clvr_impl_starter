PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_v0' \
    --cpu=1 --steps=1000 --epochs 200 --seed=0 \
    --savedir="./logs/reward_model/step-999.pth"

PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_v0' \
    --cpu=1 --steps=1000 --epochs 200 --seed=1 \
    --savedir="./logs/reward_model/step-999.pth"

PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_v0' \
    --cpu=1 --steps=1000 --epochs 200 --seed=2 \
    --savedir="./logs/reward_model/step-999.pth"
