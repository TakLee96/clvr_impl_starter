import time
import torch
from sprites_datagen.moving_sprites import MovingSpriteDataset
from sprites_datagen.rewards import *
from model import RewardPredictor
from general_utils import AttrDict

def training_loop(epoch=10):
    """ train for specified steps """
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=2,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )
    dataset = MovingSpriteDataset(spec)
    training_data = torch.utils.data.DataLoader(dataset, batch_size=1)

    net = RewardPredictor(T=3, K=4)
    # print(net)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    for ep in range(epoch):
        t = time.time()
        for _, trajectory in enumerate(training_data):
            # trajectory.images (1, 30, 3, 128, 128)
            # trajectory.rewards { 'reward_name': tensor(shape=(1, 30)) }
            # trajectory.states (1, 30, 2, 2)
            # TODO: design decision; how to handle first 2 frames?
            for i in range(spec.max_seq_len - 4):
                inputs = trajectory["images"][0, i:i+3, ...]

                labels = torch.stack([ trajectory["rewards"][r.NAME][0,i+2:i+5] for r in spec.rewards ], dim=1)

                optimizer.zero_grad()
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        print(f"[{time.time() - t:.3f}s] epoch={ep} loss={loss.item():.5f}")
