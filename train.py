import cv2
import time
import torch
import pathlib
from sprites_datagen.moving_sprites import MovingSpriteDataset
import sprites_datagen.rewards as R
from model import RewardPredictor, StateDecoder
from general_utils import AttrDict


def visualize_decoder(
    shape_per_traj=2,
    encoder_savedir="./logs/reward_model/step-999.pth",
    decoder_savedir="./logs/decode_model/step-399.pth",
    savedir="./logs/vis_dataset/",
    rewards=["AgentXReward", "AgentYReward", "TargetXReward", "TargetYReward"]
):
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=shape_per_traj,
        rewards=[ getattr(R, r) for r in rewards ],
        batch_size=1,
    )
    dataset = MovingSpriteDataset(spec)
    evaluation_data = torch.utils.data.DataLoader(dataset, batch_size=1)
    rewards = [ r.NAME for r in spec.rewards ]

    encoder = RewardPredictor(rewards)
    encoder.load_state_dict(torch.load(encoder_savedir))
    print(f"{encoder_savedir} loaded")
    for param in encoder.parameters():
        param.requires_grad = False
    decoder = StateDecoder()
    decoder.load_state_dict(torch.load(decoder_savedir))
    print(f"{decoder_savedir} loaded")
    for param in decoder.parameters():
        param.requires_grad = False

    trajectory = next(iter(evaluation_data))
    images = trajectory["images"] # B, L, 3, R, R
    output = decoder(encoder(images)[0])
    for i in range(30):
        imgfmt = lambda img: (122.5 * (img.numpy() + 1.)).transpose(1, 2, 0)
        
        cv2.imwrite(savedir + f"{i}_origin.jpg", imgfmt(images[0][i]))
        cv2.imwrite(savedir + f"{i}_decode.jpg", imgfmt(output[0][i]))
        print(f"{i}")
    print(f"images written to {savedir}")


def train_decoder(
    steps=10000,
    steps_per_save=100,
    batch_size=64,
    shape_per_traj=2,
    encoder_savedir="./logs/reward_model/step-999.pth",
    savedir="./logs/decode_model/",
    rewards=["AgentXReward", "AgentYReward", "TargetXReward", "TargetYReward"]
):
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=shape_per_traj,
        rewards=[ getattr(R, r) for r in rewards ],
        batch_size=batch_size,
    )
    dataset = MovingSpriteDataset(spec)
    training_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    rewards = [ r.NAME for r in spec.rewards ]

    frozen_encoder = RewardPredictor(rewards)
    frozen_encoder.load_state_dict(torch.load(encoder_savedir))
    print(f"{encoder_savedir} loaded")
    for param in frozen_encoder.parameters():
        param.requires_grad = False
    
    net = StateDecoder()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999)
    )

    try:
        for step in range(steps):
            t = time.time()
            trajectory = next(iter(training_data))
            inputs = trajectory["images"]   # (B, L, 3, R, R)

            optimizer.zero_grad()
            y = frozen_encoder(inputs)[0]  # (B, L, H)
            outputs = net(y) # (B, L, 3, R, R)

            loss = torch.nn.functional.mse_loss(outputs, inputs)
            loss.backward()
            optimizer.step()

            print(f"[{time.time() - t:.3f}s] step={step} loss={loss.item():.5f}")
            if step % steps_per_save == (steps_per_save - 1):
                torch.save(net.state_dict(), savedir + f"step-{step}.pth")
                print("saved model to " + savedir + f"step-{step}.pth")
    except KeyboardInterrupt:
        torch.save(net.state_dict(), savedir + f"step-{step}.pth")
        print("saved model to " + savedir + f"step-{step}.pth")


def eval_encoder(
    shape_per_traj=2,
    savedir="./logs/reward_model/step-999.pth",
    rewards=["AgentXReward", "AgentYReward", "TargetXReward", "TargetYReward"]
):
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=shape_per_traj,
        rewards=[ getattr(R, r) for r in rewards ],
        batch_size=1,
    )
    dataset = MovingSpriteDataset(spec)
    evaluation_data = torch.utils.data.DataLoader(dataset)
    rewards = [ r.NAME for r in spec.rewards ]

    net = RewardPredictor(rewards)
    net.load_state_dict(torch.load(savedir))
    print(f"{savedir} loaded")

    trajectory = next(iter(evaluation_data))
    inputs = trajectory["images"]
    raw_labels = trajectory["rewards"]
    labels = torch.stack([ raw_labels[r] for r in rewards ])
    raw_outputs = net(inputs)[-1]
    outputs = torch.stack([ raw_outputs[r] for r in rewards ])
    loss = torch.nn.functional.mse_loss(outputs, labels)
    print("outputs:", outputs)
    print("labels:", labels)
    print("loss:", loss.item())


def visualize_dataset(
    shape_per_traj=2,
    savedir="./logs/vis_dataset/",
    rewards=["AgentXReward", "AgentYReward", "TargetXReward", "TargetYReward"]
):
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

    spec = AttrDict(
        resolution=64,  # R
        max_seq_len=30, # L
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=shape_per_traj,
        rewards=[ getattr(R, r) for r in rewards ],
        batch_size=1
    )
    dataset = MovingSpriteDataset(spec)
    training_data = torch.utils.data.DataLoader(dataset, batch_size=1)
    rewards = [ r.NAME for r in spec.rewards ]

    trajectory = next(iter(training_data))
    images = trajectory["images"][0] # B, L, 3, R, R
    labels = [ trajectory["rewards"][r][0] for r in rewards ]
    for i in range(30):
        image = ((images[i].numpy() + 1.) * 122.5)
        suffix = "_".join(["{:.3f}".format(labels[j][i]) for j in range(len(rewards))])
        name = savedir + f"{i}_" + suffix + ".jpg"
        cv2.imwrite(name, image.transpose(1, 2, 0))
        print(f"{name} written")


def train_encoder(
    steps=1000,
    steps_per_save=100,
    batch_size=64,
    shape_per_traj=2,
    savedir="./logs/reward_model/",
    rewards=["AgentXReward", "AgentYReward", "TargetXReward", "TargetYReward"]
):
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

    spec = AttrDict(
        resolution=64,  # R
        max_seq_len=30, # L
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=shape_per_traj,
        rewards=[ getattr(R, r) for r in rewards ],
        batch_size=batch_size
    )
    dataset = MovingSpriteDataset(spec)
    training_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    rewards = [ r.NAME for r in spec.rewards ]

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
            raw_outputs = net(inputs)[-1]  # (K, B, L)
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
