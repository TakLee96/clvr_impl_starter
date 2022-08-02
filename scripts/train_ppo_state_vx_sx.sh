PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v0' --cpu=1 --steps=1000 --epochs=200 --seed=0

PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v0' --cpu=1 --steps=1000 --epochs=200 --seed=1

PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v0' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v0' --cpu=1 --steps=1000 --epochs=200 --seed=2


PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v1' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v1' --cpu=1 --steps=1000 --epochs=200 --seed=0

PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v1' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v1' --cpu=1 --steps=1000 --epochs=200 --seed=1

PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v1' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v1' --cpu=1 --steps=1000 --epochs=200 --seed=2


PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v2' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v2' --cpu=1 --steps=1000 --epochs=200 --seed=0

PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v2' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v2' --cpu=1 --steps=1000 --epochs=200 --seed=1

PYTHONPATH=$(pwd) python spinup/ppo.py --env='SpritesState-v2' \
    --model='MLPActorCritic' \
    --exp_name='ppo_state_v2' --cpu=1 --steps=1000 --epochs=200 --seed=2
