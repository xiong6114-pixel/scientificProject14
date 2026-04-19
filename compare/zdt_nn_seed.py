from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / "zdt_nnseed_ckpts"


class ZDTSeedNet(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def _teacher_pareto_decisions(prob_id: int, dim: int, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    x = np.zeros((t.size, dim), dtype=np.float32)
    x[:, 0] = t
    return x


def _cache_path(prob_id: int, dim: int) -> Path:
    return CACHE_DIR / f"zdt{prob_id}_nnseed_d{dim}.pt"


def _train_zdt_seed_model(
    prob_id: int,
    dim: int,
    device: str,
    train_seed: int,
    n_train: int = 4096,
    epochs: int = 250,
    batch_size: int = 256,
) -> ZDTSeedNet:
    rng = np.random.RandomState(train_seed)
    t_train = rng.rand(n_train, 1).astype(np.float32)
    y_train = _teacher_pareto_decisions(prob_id, dim, t_train[:, 0])

    x_t = torch.from_numpy(t_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)

    model = ZDTSeedNet(out_dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        order = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            xb = x_t[idx]
            yb = y_t[idx]

            pred = model(xb)
            loss_x1 = torch.mean((pred[:, :1] - yb[:, :1]) ** 2)
            loss_tail = torch.mean((pred[:, 1:] - yb[:, 1:]) ** 2) if dim > 1 else pred.sum() * 0.0
            loss = loss_x1 + 4.0 * loss_tail

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


def load_or_train_zdt_seed_model(
    prob_id: int,
    dim: int,
    device: str = "cpu",
    train_seed: int = 42,
) -> ZDTSeedNet:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = _cache_path(prob_id, dim)

    model = ZDTSeedNet(out_dim=dim).to(device)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if int(ckpt.get("prob_id", -1)) == prob_id and int(ckpt.get("dim", -1)) == dim:
            model.load_state_dict(ckpt["model"])
            model.eval()
            return model

    model = _train_zdt_seed_model(
        prob_id=prob_id,
        dim=dim,
        device=device,
        train_seed=train_seed,
    )
    torch.save(
        {
            "prob_id": int(prob_id),
            "dim": int(dim),
            "model": model.state_dict(),
        },
        ckpt_path,
    )
    return model


def make_zdt_nn_seed_builder(
    prob_id: int,
    dim: int,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    device: str = "cpu",
    train_seed: int = 42,
    tail_noise_std: float | None = None,
) -> Callable[..., np.ndarray]:
    lb_vec = np.asarray(lb_vec, dtype=np.float64).reshape(-1)
    ub_vec = np.asarray(ub_vec, dtype=np.float64).reshape(-1)
    if lb_vec.size != dim or ub_vec.size != dim:
        raise ValueError("lb_vec / ub_vec size must match dim")

    model_box: dict[str, ZDTSeedNet | None] = {"model": None}
    noise_std_default = 0.01 if prob_id != 4 else 0.03
    noise_std = float(noise_std_default if tail_noise_std is None else tail_noise_std)

    def builder_fn(
        num_needed: int,
        dim_runtime: int,
        lb_runtime: np.ndarray,
        ub_runtime: np.ndarray,
        context: dict | None = None,
    ) -> np.ndarray:
        del lb_runtime, ub_runtime, context
        if int(num_needed) <= 0:
            return np.empty((0, dim), dtype=np.float64)
        if int(dim_runtime) != dim:
            raise ValueError(f"ZDT nn-seed dim mismatch: expected {dim}, got {dim_runtime}")

        if model_box["model"] is None:
            model_box["model"] = load_or_train_zdt_seed_model(
                prob_id=prob_id,
                dim=dim,
                device=device,
                train_seed=train_seed,
            )

        model = model_box["model"]
        num_candidates = max(int(num_needed) * 4, 32)
        t = np.linspace(0.0, 1.0, num_candidates, dtype=np.float32).reshape(-1, 1)
        if num_candidates > 1:
            step = 1.0 / float(num_candidates - 1)
            jitter = (np.random.rand(num_candidates, 1).astype(np.float32) - 0.5) * step
            t = np.clip(t + jitter, 0.0, 1.0)

        with torch.no_grad():
            x = model(torch.from_numpy(t).to(device)).detach().cpu().numpy().astype(np.float64)

        x[:, 0] = np.clip(x[:, 0], lb_vec[0], ub_vec[0])
        if dim > 1:
            x[:, 1:] += np.random.normal(0.0, noise_std, size=x[:, 1:].shape)
            x[:, 1:] = np.clip(x[:, 1:], lb_vec[1:], ub_vec[1:])

        x = np.clip(x, lb_vec, ub_vec)
        x = np.unique(np.round(x, 10), axis=0)
        return x[: int(num_needed)]

    return builder_fn
