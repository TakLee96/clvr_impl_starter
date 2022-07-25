import numpy as np
import torch

class ImageEncoder(torch.nn.Module):
    def __init__(self, R=64, dim_projection=64):
        """ R: input image resolution """
        super().__init__()
        self.R = R

        # strided convolution until RxR --> 1x1
        C = 4
        layers = [
            torch.nn.Conv2d(3, C, kernel_size=3, stride=2),
            torch.nn.ReLU()
        ]
        R = int(np.floor((R - 3) / 2 + 1))
        while R > 1:
            layers.append(torch.nn.Conv2d(C, C * 2, 3, 2))
            layers.append(torch.nn.ReLU())
            C = C * 2
            R = int(np.floor((R - 3) / 2 + 1))
        self.conv_layers = torch.nn.Sequential(*layers)

        self.C = C
        self.projection = torch.nn.Linear(C, dim_projection)
    
    def forward(self, s):
        """ image state representation s: (dim_batch, 3, R, R)
            output state representation: (dim_batch, dim_projection)
        """
        assert s.shape[1:] == (3, self.R, self.R), s.shape
        dim_batch = s.shape[0]

        z = self.conv_layers(s)

        assert z.shape == (dim_batch, self.C, 1, 1), z.shape
        return self.projection(z.view([dim_batch, self.C]))


class MLP(torch.nn.Module):
    def __init__(self, dim_input, dim_output, n_layers=3, dim_hidden_units=32):
        super().__init__()
        H = dim_hidden_units

        layers = [ torch.nn.Linear(dim_input, H), torch.nn.ReLU() ]
        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(H, H))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(H, dim_output))

        self.mlp = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class HistoryEncoder(torch.nn.Module):
    def __init__(self, T=3, dim_state=64, dim_hidden=64):
        """ T: number of history states to encode """
        super().__init__()
        self.T = T
        self.dim_state = dim_state
        self.mlp = MLP(T * dim_state, dim_hidden)

    def forward(self, s):
        assert s.shape == (self.T, self.dim_state), s.shape
        return self.mlp(torch.flatten(s))

class LSTMPredictor(torch.nn.Module):
    def __init__(self, T, K, dim_hidden=64):
        """ T: rollout time steps
            K: number of tasks to model reward
        """
        super().__init__()
        self.T = T
        self.lstm = torch.nn.LSTM(dim_hidden, dim_hidden, num_layers=1)
        self.mlpk = MLP(dim_hidden, K)

    def forward(self, z):
        """ z: (64,)
            r: (T, K)
        """
        y, h = self.lstm(z[None,None,:])
        rewards = [ self.mlpk(torch.flatten(y)) ]
        for _ in range(self.T - 1):
            y, h = self.lstm(y, h)
            rewards.append(self.mlpk(torch.flatten(y)))
        return torch.stack(rewards)

class RewardPredictor(torch.nn.Module):
    def __init__(self, T, K):
        """ T: rollout time steps
            K: number of tasks to model reward
        """
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.history_encoder = HistoryEncoder()
        self.lstm_predictor = LSTMPredictor(T, K)
    
    def forward(self, x):
        """ x: images (B, C, H, W) """
        x = self.image_encoder(x)      # (B, 64)
        x = self.history_encoder(x)    # (64,)
        r_hat = self.lstm_predictor(x) # (T, K)
        return r_hat

if __name__ == '__main__':
    # debug: (T_history, C, H, W)
    inputs = torch.zeros((3, 3, 64, 64))
    net = RewardPredictor(T=3, K=4)
    outputs = net(inputs)
    print(outputs)
