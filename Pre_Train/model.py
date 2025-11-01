# models.py
import torch
import torch.nn as nn

class QStateActionFusion(nn.Module):
    def __init__(
        self,
        s_dim=513,
        a_dim=54,
        s_hidden=(1024, 512),          # 状态支路
        a_emb_dim=128,                 # 动作 one-hot -> 稠密嵌入维度
        head_hidden=(1024, 512, 256, 128)  # 融合后的 4 层隐藏层
    ):
        super().__init__()
        # 状态编码器
        s_layers, last = [], s_dim
        for h in s_hidden:
            s_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.state_net = nn.Sequential(*s_layers)    # -> (B, last)
        self.s_out_dim = last

        # 动作“嵌入”：one-hot 直接线性变换即可
        self.action_emb = nn.Linear(a_dim, a_emb_dim, bias=False)  # -> (B, a_emb_dim)

        # 融合后的 head（4 层隐藏层）
        layers, in_dim = [], self.s_out_dim + a_emb_dim
        for h in head_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, s, a):
        s_feat = self.state_net(s)         # (B, s_out_dim)
        a_feat = self.action_emb(a)        # (B, a_emb_dim)
        x = torch.cat([s_feat, a_feat], dim=-1)
        return self.head(x).squeeze(-1)    # (B,)
