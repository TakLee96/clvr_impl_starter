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

from train import training_loop as train, evaluate as eval

def vis():
    """ visualize representation training results
        by default reads from "logs/repr"
    """
    pass

def rl():
    """ oracle, image-pretrained, image-scratch """
    pass

if __name__ == '__main__':
    fire.Fire()
