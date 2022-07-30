PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v0' --cpu=1 --steps=1000