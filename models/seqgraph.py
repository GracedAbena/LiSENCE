
import torch
from torch import nn
import torch.nn.functional as F

class SequenceGraph(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=8)
        self.query = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.key = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.value = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.softmax = nn.Sigmoid()#nn.Softmax(dim=-1), got 0.8672 using softmax
        self.relu = nn.ReLU()

    def forward(self, x):# [N,C,S]
        x = self.conv(x) # [N,C,S]
        x = self.relu(x)
        N,C,S = x.shape
        x = x.permute(0,2,1)
        q = self.query(x)
        k = self.key(x)
        attent_weight = self.softmax(torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5))
        x = torch.bmm(attent_weight, self.value(x)).permute(0,2,1) 
        return x



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = torch.randn(1,128,42)
    sg = SequenceGraph(128, 16)
    out = sg.forward(x).detach()
    print(out)
    print(out.shape)
    # plt.matshow(out[0].numpy())
    # plt.show()