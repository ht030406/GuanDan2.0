import torch, torch.nn as nn, torch.optim as optim
from dataset_and_load import dataloader
from model import QStateActionFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds, train_loader = dataloader("val_dataset/1.npy", batch_size=2048, shuffle=True, num_workers=0)

# 选择模型：
ckpt = torch.load("pre_model_12.pth", map_location=device)
model = QStateActionFusion().to(device)
model.load_state_dict(ckpt["state_dict"])

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
grad_clip = 1.0
epochs = 1
use_reward_target = False   # 先蒸馏/预训练到 q，再视情况切 True 拟合 r

model.eval()
for ep in range(1, epochs+1):
    tot = 0.0
    for batch in train_loader:
        s = batch["s"].to(device)
        a = batch["a"].to(device)
        r = batch["r"].to(device).squeeze(-1)
        q_teacher = batch["q"].to(device).squeeze(-1)

        q_pred = model(s, a)                                 # == network(concate(s,a))
        target = r if use_reward_target else q_teacher
        loss = criterion(q_pred, target)

        optimizer.zero_grad()
        # loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # optimizer.step()
        tot += loss.item() * s.size(0)

    print(f"Epoch {ep}: loss={tot/len(ds):.6f}")