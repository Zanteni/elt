import torch
import torch.nn as nn

from utils import build_linear_warmup_cosine_schedule


# dummy model
model = nn.Linear(10, 10)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


cfg = {
    "scheduler": {
        "warmup_steps": 5000,
        "min_lr_ratio": "1e-7",   # simulate your YAML issue
    },
    "train": {
        "total_steps": 200000
    }
}


scheduler = build_linear_warmup_cosine_schedule(
    optimizer,
    cfg
)


print("min_lr_ratio type test passed")

for step in [0, 1000, 5000, 10000, 200000]:
    scheduler.last_epoch = step
    lr = scheduler.get_last_lr()[0]
    print(f"step {step}: lr={lr}")