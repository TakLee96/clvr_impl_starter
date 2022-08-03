PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v0_c4_s4000' --cpu=4 --steps=4000 --epochs 500
