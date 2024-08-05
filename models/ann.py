
import torch
from torch import nn
from models import attention

class Fully(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.lin = nn.Linear(in_feat, out_feat)
        self.act = nn.Mish(True)
        self.bn = nn.Identity()

        nn.init.xavier_normal_(self.lin.weight, gain=1.0)
        nn.init.constant_(self.lin.bias, 0.)

    def forward(self, x):
        return self.bn(self.act(self.lin(x)))

class FingerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 256
        inter = hidden//2

        self.macc_l1 = Fully(167, hidden)
        self.erg_l1 = Fully(441, hidden)
        self.pub_l1 = Fully(881, hidden)
        self.mol2vec_l1 = Fully(300, hidden)

        self.macc_l2 = Fully(hidden, inter)
        self.erg_l2 = Fully(hidden, inter)
        self.pub_l2 = Fully(hidden, inter)
        self.mol2vec_l2 = Fully(hidden, inter)

        self.atten = attention.MultiAttention([3,5])
        self.linear = nn.Sequential(nn.Linear(inter*4, 256), nn.Mish())

    def forward(self, macc_x, erg_x, pub_x, mol2vec_x):# x:[N,C]
        
        macc_x = self.macc_l1(macc_x)
        erg_x = self.erg_l1(erg_x)
        pub_x = self.pub_l1(pub_x)
        mol2vec_x = self.mol2vec_l1(mol2vec_x)

        macc_x = self.macc_l2(macc_x)
        erg_x = self.erg_l2(erg_x)
        pub_x = self.pub_l2(pub_x)
        mol2vec_x = self.mol2vec_l2(mol2vec_x)

        out = torch.concat([macc_x, erg_x, pub_x, mol2vec_x], dim=-1)
        out = self.atten(out)
        out = self.linear(out)
        return out

class FingerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = FingerEncoder()
        self.fc = nn.Linear(256, 1)
    
    def forward(self, macc_x, erg_x, pub_x, mol2vec_x):
        x = self.encoder(macc_x, erg_x, pub_x, mol2vec_x)
        x = self.fc(x).reshape(-1)
        return x


if __name__ == "__main__":

    net = FCATTEN(2)
    macc_x, erg_x, pub_x = torch.ones(4, 167), torch.ones(4, 441), torch.ones(4, 881)
    out = net(macc_x, erg_x, pub_x)
    print(out.shape)
    print(out[0])

