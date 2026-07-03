"""Training loop for the terrain-conditioned expansion diffusion model."""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import UNet, Diffusion, EMA


class ExpansionDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.cond = torch.from_numpy(d["cond"])
        self.target = torch.from_numpy(d["target"])

    def __len__(self):
        return len(self.cond)

    def __getitem__(self, i):
        return self.cond[i], self.target[i]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ExpansionDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=2 if device == "cuda" else 0, drop_last=True)
    model = UNet(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device} samples={len(ds)} params={n_params/1e6:.1f}M")
    diff = Diffusion(model, device)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(len(dl), 1))
    os.makedirs(args.out, exist_ok=True)
    history = []
    step = 0
    for epoch in range(args.epochs):
        t0, running = time.time(), 0.0
        for cond, target in dl:
            cond, target = cond.to(device), target.to(device)
            loss = diff.loss(target, cond)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            running += loss.item()
            step += 1
        avg = running / max(len(dl), 1)
        history.append(avg)
        print(f"[train] epoch {epoch+1}/{args.epochs} loss={avg:.4f} "
              f"({time.time()-t0:.0f}s)")
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            ema_model = UNet(base=args.base).to(device)
            ema.copy_to(ema_model)
            torch.save({"model": model.state_dict(),
                        "ema": ema_model.state_dict(),
                        "history": history, "args": vars(args)},
                       os.path.join(args.out, "ckpt.pt"))
            print(f"[train] checkpoint saved at epoch {epoch+1}")
    np.savetxt(os.path.join(args.out, "loss_history.txt"), np.array(history))
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--ckpt_every", type=int, default=20)
    train(ap.parse_args())
