PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v2' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v2' --cpu=1 --steps=1000 --epochs 100
