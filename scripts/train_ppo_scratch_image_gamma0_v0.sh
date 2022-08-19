PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_scratch_image_gamma0_v0' \
    --cpu=1 --steps=4000 --epochs 500 --seed=0 --gamma=0
