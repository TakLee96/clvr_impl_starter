PYTHONPATH=$(pwd) python spinup/ppo.py --env='Sprites-v0' \
    --model='ImageActorCritic' \
    --exp_name='ppo_scratch_image_v0' \
    --cpu=2