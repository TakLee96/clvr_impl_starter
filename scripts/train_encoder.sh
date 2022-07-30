python main.py train_encoder \
    --shape_per_traj=2 --batch_size=32 \
    --savedir="./logs/reward_model/" \
    --rewards="AgentXReward","AgentYReward","TargetXReward","TargetYReward"
