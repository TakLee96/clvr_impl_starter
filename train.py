import time
import torch
import pathlib
from sprites_datagen.moving_sprites import MovingSpriteDataset
from sprites_datagen.rewards import *
from model import RewardPredictor
from general_utils import AttrDict


def evaluate(
    savedir="./logs/reward_model/step-999.pth"
):
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=2,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
        batch_size=1,
    )
    dataset = MovingSpriteDataset(spec)
    evaluation_data = torch.utils.data.DataLoader(dataset)
    rewards = [ r.NAME for r in spec.rewards ]

    net = RewardPredictor(rewards)
    net.load_state_dict(torch.load(savedir))
    trajectory = next(iter(evaluation_data))
    inputs = trajectory["images"]
    raw_labels = trajectory["rewards"]
    labels = torch.stack([ raw_labels[r] for r in rewards ])
    raw_outputs = net(inputs)
    outputs = torch.stack([ raw_outputs[r] for r in rewards ])
    loss = torch.nn.functional.mse_loss(outputs, labels)
    print("outputs:", outputs)
    print("labels:", labels)
    print("loss:", loss.item())


def training_loop(
    steps=1000,
    steps_per_save=100,
    batch_size=64,
    savedir="./logs/reward_model/"
):
    """ train for specified steps """
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

    spec = AttrDict(
        resolution=64,  # R
        max_seq_len=30, # L
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=2,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
        batch_size=batch_size
    )
    dataset = MovingSpriteDataset(spec)
    training_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    rewards = [ r.NAME for r in spec.rewards ]

    # import cv2
    # trajectory = next(iter(training_data))
    # images = trajectory["images"][0] # B, L, 3, R, R
    # labels = [ trajectory["rewards"][r][0] for r in rewards ]
    # for i in range(30):
    #     image = ((images[i].numpy() + 1.) * 122.5)
    #     label = (labels[0][i], labels[1][i], labels[2][i], labels[3][i])
    #     name = f"vis/{i}_" + "{:.3f}_{:.3f}_{:.3f}_{:.3f}.jpg".format(*label)
    #     cv2.imwrite(name, image.transpose(1, 2, 0))
    #     print(f"{name} written")
    # exit()

    net = RewardPredictor(rewards)
    print(net)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999)
    )

    try:
        for step in range(steps):
            t = time.time()
            trajectory = next(iter(training_data))
            # trajectory.states (1, 30, 2, 2)
            inputs = trajectory["images"]   # (B, L, 3, R, R)
            raw_labels = trajectory["rewards"] # dict(K) => (B, L)
            labels = torch.stack([ raw_labels[r] for r in rewards ])   # (K, B, L)

            optimizer.zero_grad()
            raw_outputs = net(inputs)  # (K, B, L)
            outputs = torch.stack([ raw_outputs[r] for r in rewards ]) # (K, B, L)
            # assert labels.shape == outputs.shape, f"{labels} {outputs}"

            loss = torch.nn.functional.mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"[{time.time() - t:.3f}s] step={step} loss={loss.item():.5f}")
            if step % steps_per_save == (steps_per_save - 1):
                torch.save(net.state_dict(), savedir + f"step-{step}.pth")
                print("saved model to " + savedir + f"step-{step}.pth")
    except KeyboardInterrupt:
        torch.save(net.state_dict(), savedir + f"step-{step}.pth")
        print("saved model to " + savedir + f"step-{step}.pth")
