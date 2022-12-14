import numpy as np
import torch


class StateDecoder(torch.nn.Module):
    def __init__(self, input_size=64, output_size=64, hidden_size=128):
        super().__init__()
        self.output_size = output_size
        self.mlp = MLP(
            input_size=input_size,
            output_size=output_size * output_size,
            n_layers=4,
            hidden_size=hidden_size
        )

    def forward(self, x):
        """ x: (B, T, C) """
        B = x.shape[0]
        T = x.shape[1]
        R = self.output_size

        x = self.mlp(x)
        x = x.view(B, T, 1, R, R)
        return x.expand(B, T, 3, R, R) - 1


class ImageEncoder(torch.nn.Module):
    def __init__(self, input_size=64, output_size=64):
        super().__init__()
        self.R = input_size

        # strided convolution until RxR --> 1x1
        C = 4
        layers = [
            torch.nn.Conv2d(3, C, kernel_size=3, stride=2),
            torch.nn.ReLU()
        ]
        R = int(np.floor((input_size - 3) / 2 + 1))
        while R > 1:
            layers.append(torch.nn.Conv2d(C, C * 2, 3, 2))
            layers.append(torch.nn.ReLU())
            C = C * 2
            R = int(np.floor((R - 3) / 2 + 1))
        self.conv_layers = torch.nn.Sequential(*layers)

        self.C = C
        self.projection = torch.nn.Linear(C, output_size)
    
    def forward(self, s):
        """ image state representation s: (B, T, 3, R, R)
            @return (B, T, O)
        """
        B = s.shape[0]
        T = s.shape[1]
        # assert s.shape[2:] == (3, self.R, self.R), s.shape

        s = s.view(B * T, 3, self.R, self.R)
        z = self.conv_layers(s)  # (B * T, C, 1, 1)
        return self.projection(z.view([B, T, self.C]))


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, n_layers=3, hidden_size=32):
        super().__init__()
        H = hidden_size

        layers = [ torch.nn.Linear(input_size, H), torch.nn.ReLU() ]
        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(H, H))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(H, output_size))

        self.mlp = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class LSTMPredictor(torch.nn.Module):
    def __init__(self, rewards, input_size=64, hidden_size=64, use_gpu=False):
        """ rewards: list of rewards """
        super().__init__()
        self.hidden_size = hidden_size
        self.h_init = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.c_init = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.mlpk = torch.nn.ModuleDict({ r: MLP(hidden_size, 1) for r in rewards })
        self.use_gpu = use_gpu

    def forward(self, x, hc=None):
        """ B=batch T=time I=input H=hidden
            x: (B, T, I)
            hc: tuple(h, c)
                h: (1, B, H)
                c: (1, B, H)
            @return
                y: (B, T, H)
                hc: same as input
                { "reward_name": (B, T, 1) }
        """
        B = x.shape[0]
        if hc is None:
            h = self.h_init.expand(1, B, -1)
            c = self.c_init.expand(1, B, -1)
            hc = (h, c)

        if self.use_gpu:
            hc = (hc[0].contiguous(), hc[1].contiguous())

        y, hc = self.lstm(x, hc) # (T, 64)
        output_dict = {}
        for r, mlp in self.mlpk.items():
            output = mlp(y)
            # assert output.shape[-1] == 1, output.shape
            output_dict[r] = output.view(output.shape[:-1])
        return y, hc, output_dict


class RewardPredictor(torch.nn.Module):
    def __init__(self, rewards, use_gpu=False):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.lstm_predictor = LSTMPredictor(rewards, use_gpu=use_gpu)
    
    def forward(self, x, hc=None):
        """ x: images (B, T, C, H, W) """
        z = self.image_encoder(x)
        return self.lstm_predictor(z, hc)


if __name__ == '__main__':
    # TODO: debug network shapes
    from sprites_datagen.rewards import *
    inputs = torch.zeros((1, 3, 3, 64, 64))
    rewards = [AgentXReward, AgentYReward, TargetXReward, TargetYReward]
    rewards = [ r.NAME for r in rewards ]
    net = RewardPredictor(rewards)
    print(net)
    outputs = net(inputs)
    print(outputs)
