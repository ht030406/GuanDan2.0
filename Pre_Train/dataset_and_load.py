# dataset_and_loader.py
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NpyTransitionDataset(Dataset):
    def __init__(self, pattern: str):
        self.items = []  # 每条是 (s, a, r, q)
        for f in sorted(glob.glob(pattern)):
            obj = np.load(f, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.shape == ():  # 0-d array 包了个 dict
                obj = obj.item()
            assert isinstance(obj, dict), f"{f} 不是 dict 格式"

            # 转成 float32 / 形状对齐
            s = np.asarray(obj["s"], dtype=np.float32)
            a = np.asarray(obj["a"], dtype=np.float32)  # 你截图里是 0/1 的 one-hot，转成 float 即可
            r = np.asarray(obj["r"], dtype=np.float32).reshape(-1, 1)
            q = np.asarray(obj["q"], dtype=np.float32).reshape(-1, 1)

            n = min(len(s), len(a), len(r), len(q))  # 以防各字段长度略有出入
            s, a, r, q = s[:n], a[:n], r[:n], q[:n]

            self.items.append((
                torch.from_numpy(s),
                torch.from_numpy(a),
                torch.from_numpy(r),
                torch.from_numpy(q),
            ))

        # 预先 concat 到一起，__getitem__ 简单高效
        self.s = torch.cat([t[0] for t in self.items], dim=0)
        self.a = torch.cat([t[1] for t in self.items], dim=0)
        self.r = torch.cat([t[2] for t in self.items], dim=0)
        self.q = torch.cat([t[3] for t in self.items], dim=0)

        # 记录维度，给网络初始化用
        self.s_dim = self.s.shape[1]
        self.a_dim = self.a.shape[1]

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return {
            "s": self.s[idx],
            "a": self.a[idx],
            "r": self.r[idx],
            "q": self.q[idx],
        }

def dataloader(pattern="./*.npy", batch_size=256, shuffle=True, num_workers=0):
    ds = NpyTransitionDataset(pattern)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return ds, dl
