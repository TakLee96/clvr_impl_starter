PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v2' \
    --model='ImageActorCritic' \
    --exp_name='ppo_scratch_image_v2' \
    --cpu=1 --steps=1000 --epochs 500
