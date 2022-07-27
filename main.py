"""
@author Jiahang Li (jiahangl@usc.edu)

Design:
main.py - fire commands
ppo.py  - PPO implementation

Goal:
1. Implement reward-induced representation learning model
    - Train the model using provided dataset
2. Visualize training results
    - Verify model trained correctly (decoder network?)
3. Implement an RL algorithm of your choice to follow target
    - Verify implementation by oracle model first
4. Train image-based RL agent using pre-trained encoder
    - Compare learning curve to image-scratch
5. Plot multiple curves
"""

import fire

from train import (
    train_encoder,
    eval_encoder,
    train_decoder,
    visualize_decoder,
    visualize_dataset
)

if __name__ == '__main__':
    fire.Fire()
