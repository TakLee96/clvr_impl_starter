python main.py train_encoder --steps=200 \
    --shape_per_traj=2 \
    --savedir="./logs/reward_model/" \
    --rewards="AgentXReward","AgentYReward","TargetXReward","TargetYReward"
