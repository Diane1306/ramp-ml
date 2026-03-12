import torch.nn as nn


class TCNReset(nn.Module):
    def __init__(self, ch: int = 48, k: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, ch, k, padding=k // 2),
            nn.ReLU(),
            nn.Conv1d(ch, ch, k, padding=(k // 2) * 2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(ch, ch, k, padding=(k // 2) * 4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(ch, 1, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B, L)