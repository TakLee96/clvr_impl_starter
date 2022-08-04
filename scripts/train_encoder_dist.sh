python main.py train_encoder \
    --shape_per_traj=2 \
    --savedir=./logs/encoder_dist/ \
    --rewards=DistanceReward, \
    --steps=500
