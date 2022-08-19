python main.py train_decoder \
    --shape_per_traj=1 \
    --savedir=./logs/decoder_vpr/ \
    --encoder_savedir=./logs/encoder_vpr/step-199.pth \
    --rewards=VertPosReward, \
    --steps=500
