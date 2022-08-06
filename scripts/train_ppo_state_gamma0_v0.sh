PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 100 --gamma 0 --seed 0
