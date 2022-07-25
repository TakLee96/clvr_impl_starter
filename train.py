import time
import torch
import pathlib
from sprites_datagen.moving_sprites import MovingSpriteDataset
from sprites_datagen.rewards import *
from model import RewardPredictor
from general_utils import AttrDict


def evaluate(savedir="./logs/reward_model/step-9099.pth"):
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=2,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )
    dataset = MovingSpriteDataset(spec)
    evaluation_data = torch.utils.data.DataLoader(dataset, batch_size=1)

    net = RewardPredictor(T=3, K=4)
    net.load_state_dict(torch.load(savedir))
    criterion = torch.nn.MSELoss()

    for trajectory in evaluation_data:
        for i in range(spec.max_seq_len - 4):
            inputs = trajectory["images"][0, i:i+3, ...]
            labels = torch.stack([ trajectory["rewards"][r.NAME][0,i+2:i+5] for r in spec.rewards ], dim=1)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            print(i, outputs, labels, loss.item())


def training_loop(epoch=10, savedir="./logs/reward_model/"):
    """ train for specified steps """
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

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
    print(net)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999)
    )

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
        if ep % 1000 == 999:
            torch.save(net.state_dict(), savedir + f"step-{ep}.pth")
