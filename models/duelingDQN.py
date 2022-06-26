import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.linear1_ad = nn.Linear(7*7*64, 512)
        self.linear1_val = nn.Linear(7*7*64, 512)

        self.linear2_ad = nn.Linear(512, n_actions)
        self.linear2_val = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        ad = self.relu(self.linear1_ad(x))
        val = self.relu(self.linear1_val(x))

        ad = self.linear2_ad(ad)
        val = self.linear2_val(val)

        x = val + ad - ad.mean()

        return x