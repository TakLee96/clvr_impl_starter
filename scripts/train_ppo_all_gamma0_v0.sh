PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 100 --seed=0 --gamma=0 

PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 100 --seed=1 --gamma=0 

PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 100 --seed=2 --gamma=0 


PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_pretrained_image_gamma0_v0' \
    --freeze --cpu=1 --steps=4000 --epochs 500 --seed=0 --gamma=0 \
    --savedir="./logs/reward_model/step-999.pth"

PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_pretrained_image_gamma0_v0' \
    --freeze --cpu=1 --steps=4000 --epochs 500 --seed=1 --gamma=0 \
    --savedir="./logs/reward_model/step-999.pth"

PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_pretrained_image_gamma0_v0' \
    --freeze --cpu=1 --steps=4000 --epochs 500 --seed=2 --gamma=0 \
    --savedir="./logs/reward_model/step-999.pth"


PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=0 --gamma=0 \
    --savedir="./logs/reward_model/step-999.pth"

PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=1 --gamma=0 \
    --savedir="./logs/reward_model/step-999.pth"

PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_finetune_image_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=2 --gamma=0 \
    --savedir="./logs/reward_model/step-999.pth"


PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_scratch_image_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=0 --gamma=0

PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_scratch_image_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=1 --gamma=0

PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_scratch_image_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=2 --gamma=0
