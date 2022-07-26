python main.py train_decoder \
    --shape_per_traj=1 \
    --savedir=./logs/decoder_hpr/ \
    --encoder_savedir=./logs/encoder_hpr/step-199.pth \
    --rewards=HorPosReward, \
    --steps=500
