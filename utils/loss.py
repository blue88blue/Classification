import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_one_hot



class AMSoftMaxLoss2D(nn.Module):
    def __init__(self, inchannel, n_class, m=0.35, s=30, device=torch.device("cpu")):
        super().__init__()
        self.linear = nn.Conv2d(inchannel, n_class, kernel_size=1, bias=False)
        self.linear.to(device)
        self.n_class = n_class
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, fearure, label=None):
        assert len(fearure.size()) == 4
        # 标准化
        fearure = F.normalize(fearure, p=2, dim=1)
        self.linear.weight.data = F.normalize(self.linear.weight.data, p=2, dim=1)
        cos = self.linear(fearure)

        if label == None:
            pred = torch.softmax(cos*self.s, dim=1)
            return {"main_out": pred}
        else:
            assert len(label.size()) == 3
            onehot_label = make_one_hot(label, self.n_class)  # (b, n_class, -1)
            cos_m = cos - onehot_label * self.m
            cos_m_s = cos_m * self.s
            loss = self.ce(cos_m_s, label)
            return {"loss": loss}





class AMSoftMaxLoss(nn.Module):
    def __init__(self, inchannel, n_class, m=0.35, s=30, device=torch.device("cpu")):
        super().__init__()
        self.linear = nn.Linear(inchannel, n_class, bias=False)
        self.linear.to(device)
        self.n_class = n_class
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, fearure, label=None):
        assert len(fearure.size()) == 2
        # 标准化
        fearure = F.normalize(fearure, p=2, dim=1)
        self.linear.weight.data = F.normalize(self.linear.weight.data, p=2, dim=1)
        cos = self.linear(fearure)  # (b, n_class)

        if label == None:
            pred = torch.softmax(cos*self.s, dim=1)
            return {"main_out": pred}
        else:
            assert len(label.size()) == 2
            onehot_label = make_one_hot(label, self.n_class).squeeze(-1)  # (b, n_class)
            cos_m = cos - onehot_label * self.m
            cos_m_s = cos_m * self.s
            loss = self.ce(cos_m_s, label)
            return {"loss": loss}
