# models.py
import torch
import torch.nn as nn
import numpy as np
from Agent.message2state import action_to_state

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

    def forward(self, state, actions):
        """
        参数
        ----
        state: np.ndarray or torch.Tensor
            形状 (513,) 或 (1, 513)
        actions: list[np.ndarray] | list[torch.Tensor] | torch.Tensor
            - 若为 list，则每个元素形状 (54,)
            - 若为张量，则形状应为 (K, 54)

        返回
        ----
        best_idx: int
            最佳动作在 `actions` 列表/张量中的下标
        """

        device = next(self.parameters()).device
        dtype = torch.float32

        # --- state: numpy -> torch ---
        if isinstance(state, np.ndarray):
            s = torch.from_numpy(state).to(device=device, dtype=dtype)
        elif isinstance(state, torch.Tensor):
            s = state.to(device=device, dtype=dtype)
        else:
            s = torch.as_tensor(state, dtype=dtype, device=device)

        if s.dim() == 1:
            s = s.unsqueeze(0)  # -> (1, 513)
        elif not (s.dim() == 2 and s.size(0) == 1 and s.size(1) == self.s_out_dim or s.size(1) == 513):
            raise ValueError(f"state 形状不支持: {tuple(s.shape)}，期望 (513,) 或 (1,513)")

        # --- actions: list/tensor -> (K, 54) ---
        if isinstance(actions, torch.Tensor):
            a = actions.to(device=device, dtype=dtype)  # (K, 54)
        else:
            # list 情况
            a_list = []
            for i, act in enumerate(actions):
                if isinstance(act, np.ndarray):
                    t = torch.from_numpy(act).to(device=device, dtype=dtype)
                elif isinstance(act, torch.Tensor):
                    t = act.to(device=device, dtype=dtype)
                else:
                    t = torch.as_tensor(act, dtype=dtype, device=device)
                if t.dim() != 1:
                    raise ValueError(f"第 {i} 个 action 形状应为 (54,), 实际 {tuple(t.shape)}")
                a_list.append(t)
            a = torch.stack(a_list, dim=0)  # (K, 54)

        K = a.size(0)
        s_rep = s.repeat(K, 1)  # (K, 513)

        # --- 计算每个 Q(s, a) 并取 argmax ---
        s_feat = self.state_net(s_rep)  # (K, s_out_dim)
        a_feat = self.action_emb(a)  # (K, a_emb_dim)
        x = torch.cat([s_feat, a_feat], dim=-1)  # (K, s_out_dim + a_emb_dim)
        q = self.head(x).squeeze(-1)  # (K,)
        best_idx = int(torch.argmax(q).item())
        return best_idx


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = QStateActionFusion()
#
# dict = {"actions":[{"index": 2,"action": ["Bomb", "9", ["H9", "H9", "C9", "D9"]]},{"index": 1,"action": ["Pair", "A", ["HA", "DA"]]}, {"index": 3,"action": ["Bomb", "8", ["H8", "H8", "C8", "D8"]]}]}
# actions = dict['actions']
# action_vec = action_to_state(actions)
#
# s = np.zeros(513,int)
#
# idx = model(s, action_vec)
# print(idx)
