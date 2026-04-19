import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


N_POINTS = 100
FEAT_DIM = 49
IN_DIM = N_POINTS * FEAT_DIM

DEFAULT_LEVEL_TO_PILES = {
    0: 0,
    1: 10,
    2: 25,
    3: 40,
}


class SetTransformerPriorityCountCapacity(nn.Module):
    def __init__(
        self,
        feat_dim: int = FEAT_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # 显式关闭 nested tensor，避免 warning
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.priority_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.count_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.capacity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )

    @staticmethod
    def masked_mean_pool(h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_f = valid_mask.float().unsqueeze(-1)
        h_sum = (h * valid_f).sum(dim=1)
        denom = valid_f.sum(dim=1).clamp_min(1.0)
        return h_sum / denom

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        valid_mask = ~key_padding_mask

        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        priority_logits = self.priority_head(h).squeeze(-1)
        pooled = self.masked_mean_pool(h, valid_mask)
        count_raw = self.count_head(pooled).squeeze(-1)
        count_pred = torch.nn.functional.softplus(count_raw)
        capacity_logits = self.capacity_head(h)

        return priority_logits, count_pred, capacity_logits


def compute_pad_mask_from_raw_X_single(x_raw_flat: np.ndarray) -> np.ndarray:
    x_raw_flat = np.asarray(x_raw_flat, dtype=np.float32).reshape(-1)
    if x_raw_flat.shape != (IN_DIM,):
        raise ValueError(f"x_raw_flat 应为 [{IN_DIM}]，实际 {x_raw_flat.shape}")
    x3 = x_raw_flat.reshape(N_POINTS, FEAT_DIM)
    return np.all(x3 == 0.0, axis=-1)


def safe_sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    temperature = max(float(temperature), 1e-6)
    x = x / temperature
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / max(np.sum(ex), 1e-12)


def round_and_clip_count(count_pred: float, n_valid: int) -> int:
    k = int(np.rint(float(count_pred)))
    k = max(0, min(k, int(n_valid)))
    return k


def default_pack_solution(
    build_vec: np.ndarray,
    piles_vec: np.ndarray,
    level_vec: Optional[np.ndarray] = None,
    context: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    默认把解打包成 [build(100), piles(100)] -> [200]
    如果真实编码不是这样，需要在外部传入 pack_solution_fn 覆盖它。
    """
    build_vec = np.asarray(build_vec, dtype=float).reshape(-1)
    piles_vec = np.asarray(piles_vec, dtype=float).reshape(-1)

    if build_vec.shape[0] != N_POINTS or piles_vec.shape[0] != N_POINTS:
        raise ValueError("default_pack_solution 要求长度都是 100")

    return np.concatenate([build_vec, piles_vec], axis=0)


def sanitize_population(
    pop: np.ndarray,
    dim: int,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
) -> np.ndarray:
    pop = np.asarray(pop, dtype=float)
    if pop.size == 0:
        return np.empty((0, dim), dtype=float)

    if pop.ndim == 1:
        pop = pop.reshape(1, -1)

    if pop.shape[1] != dim:
        raise ValueError(f"个体维度不匹配，应为 {dim}，实际 {pop.shape[1]}")

    pop = np.clip(pop, lb_vec, ub_vec)
    pop = np.unique(pop, axis=0)
    return pop


def _sol_key(sol: np.ndarray) -> Tuple[float, ...]:
    sol = np.asarray(sol, dtype=float).reshape(-1)
    return tuple(np.round(sol, 8).tolist())


@dataclass
class SeedDecodeConfig:
    num_seeds: int = 12
    count_radius: int = 2
    top_margin: int = 4
    stochastic_ratio: float = 0.75
    priority_temperature: float = 1.0
    capacity_temperature: float = 1.0
    min_k: int = 1
    max_k: int = 100


class PCCSeedGenerator:
    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        level_to_piles: Optional[Dict[int, int]] = None,
    ):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"找不到 ckpt: {ckpt_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        arch = ckpt["arch"]
        self.model = SetTransformerPriorityCountCapacity(
            feat_dim=ckpt["feat_dim"],
            d_model=arch["d_model"],
            nhead=arch["nhead"],
            num_layers=arch["num_layers"],
            dim_feedforward=arch["ff_dim"],
            dropout=arch["dropout"],
        ).to(self.device)

        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_std = np.asarray(ckpt["scaler_std"], dtype=np.float32)
        self.scaler_std = np.where(np.abs(self.scaler_std) < 1e-8, 1.0, self.scaler_std)

        self.level_to_piles = level_to_piles or DEFAULT_LEVEL_TO_PILES

    @torch.no_grad()
    def predict(self, x_raw_flat: np.ndarray) -> Dict[str, np.ndarray]:
        x_raw_flat = np.asarray(x_raw_flat, dtype=np.float32).reshape(-1)
        pad_mask = compute_pad_mask_from_raw_X_single(x_raw_flat)
        valid_mask = ~pad_mask

        x_norm = (x_raw_flat[None, :] - self.scaler_mean) / self.scaler_std
        x_tensor = torch.from_numpy(x_norm).float().to(self.device).reshape(1, N_POINTS, FEAT_DIM)
        pad_tensor = torch.from_numpy(pad_mask[None, :]).bool().to(self.device)

        priority_logits_t, count_pred_t, capacity_logits_t = self.model(
            x_tensor,
            key_padding_mask=pad_tensor,
        )

        priority_logits = priority_logits_t[0].detach().cpu().numpy()
        priority_prob = safe_sigmoid_np(priority_logits)
        count_pred = float(count_pred_t[0].detach().cpu().item())
        capacity_logits = capacity_logits_t[0].detach().cpu().numpy()

        return {
            "priority_logits": priority_logits,
            "priority_prob": priority_prob,
            "count_pred": count_pred,
            "capacity_logits": capacity_logits,
            "pad_mask": pad_mask,
            "valid_mask": valid_mask,
        }

    def _sample_topk(
        self,
        scores: np.ndarray,
        candidate_indices: np.ndarray,
        k: int,
        temperature: float,
    ) -> np.ndarray:
        if k <= 0 or len(candidate_indices) == 0:
            return np.zeros((0,), dtype=np.int64)

        k = min(k, len(candidate_indices))
        probs = softmax_np(scores[candidate_indices], temperature=temperature)
        picked = np.random.choice(candidate_indices, size=k, replace=False, p=probs)
        return np.sort(picked)

    def _decode_one_solution(
        self,
        priority_logits: np.ndarray,
        count_pred: float,
        capacity_logits: np.ndarray,
        valid_mask: np.ndarray,
        k_override: Optional[int],
        stochastic: bool,
        cfg: SeedDecodeConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        if n_valid == 0:
            raise ValueError("没有有效候选点")

        base_k = round_and_clip_count(count_pred, n_valid)
        k = base_k if k_override is None else int(k_override)
        k = max(cfg.min_k, k)
        k = min(cfg.max_k, k)
        k = min(k, n_valid)

        if k <= 0:
            raise ValueError("解码得到的 k <= 0")

        sorted_valid = valid_indices[np.argsort(-priority_logits[valid_indices])]

        if not stochastic:
            chosen = np.sort(sorted_valid[:k])
        else:
            pool_size = min(n_valid, max(k + cfg.top_margin, cfg.top_margin + 1))
            candidate_pool = sorted_valid[:pool_size]
            chosen = self._sample_topk(
                scores=priority_logits,
                candidate_indices=candidate_pool,
                k=k,
                temperature=cfg.priority_temperature,
            )

        build_vec = np.zeros(N_POINTS, dtype=np.int64)
        build_vec[chosen] = 1

        level_vec = np.zeros(N_POINTS, dtype=np.int64)
        piles_vec = np.zeros(N_POINTS, dtype=np.int64)

        for j in chosen:
            logits_123 = capacity_logits[j, 1:4]
            if stochastic:
                probs_123 = softmax_np(logits_123, temperature=cfg.capacity_temperature)
                lvl = int(np.random.choice(np.array([1, 2, 3]), p=probs_123))
            else:
                lvl = int(np.argmax(logits_123)) + 1

            level_vec[j] = lvl
            piles_vec[j] = int(self.level_to_piles[lvl])

        return build_vec, level_vec, piles_vec

    def generate_packed_seed_solutions(
        self,
        x_raw_flat: np.ndarray,
        dim: int,
        lb_vec: np.ndarray,
        ub_vec: np.ndarray,
        pack_solution_fn: Optional[Callable[..., np.ndarray]] = None,
        repair_fn: Optional[Callable[..., np.ndarray]] = None,
        evaluate_fn: Optional[Callable[..., Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        decode_cfg: Optional[SeedDecodeConfig] = None,
        keep_only_valid: bool = True,
    ) -> np.ndarray:
        cfg = decode_cfg or SeedDecodeConfig()
        pack_solution_fn = pack_solution_fn or default_pack_solution

        if int(cfg.num_seeds) <= 0:
            return np.empty((0, dim), dtype=float)

        pred = self.predict(x_raw_flat)
        priority_logits = pred["priority_logits"]
        count_pred = pred["count_pred"]
        capacity_logits = pred["capacity_logits"]
        valid_mask = pred["valid_mask"]

        n_valid = int(valid_mask.sum())
        if n_valid <= 0:
            return np.empty((0, dim), dtype=float)

        base_k = round_and_clip_count(count_pred, n_valid)
        if base_k <= 0:
            base_k = min(max(cfg.min_k, 1), n_valid)

        k_list = []
        for dk in range(-cfg.count_radius, cfg.count_radius + 1):
            kk = base_k + dk
            kk = max(cfg.min_k, kk)
            kk = min(cfg.max_k, kk)
            kk = min(kk, n_valid)
            if kk > 0 and kk not in k_list:
                k_list.append(kk)

        if len(k_list) == 0:
            return np.empty((0, dim), dtype=float)

        sols = []
        seen = set()

        def try_add_solution(sol: np.ndarray) -> bool:
            sol = np.asarray(sol, dtype=float).reshape(-1)

            if repair_fn is not None:
                sol = repair_fn(sol, context)

            if sol.size != dim:
                return False

            if evaluate_fn is not None and keep_only_valid:
                obj = evaluate_fn(sol, context)
                if obj is None:
                    return False
                arr = np.asarray(obj, dtype=float).reshape(-1)
                if arr.size == 0 or (not np.all(np.isfinite(arr))) or np.any(arr == -1):
                    return False

            key = _sol_key(sol)
            if key in seen:
                return False

            seen.add(key)
            sols.append(sol)
            return True

        # 一个贪心解
        try:
            build_vec, level_vec, piles_vec = self._decode_one_solution(
                priority_logits=priority_logits,
                count_pred=count_pred,
                capacity_logits=capacity_logits,
                valid_mask=valid_mask,
                k_override=base_k,
                stochastic=False,
                cfg=cfg,
            )
            sol = pack_solution_fn(build_vec, piles_vec, level_vec, context)
            try_add_solution(sol)
        except Exception:
            pass

        # 多个扰动解：增加最大尝试次数，避免死循环
        max_tries = max(50, 20 * int(cfg.num_seeds), 5 * len(k_list))
        tries = 0

        while len(sols) < int(cfg.num_seeds) and tries < max_tries:
            made_progress = False

            for kk in k_list:
                if len(sols) >= int(cfg.num_seeds) or tries >= max_tries:
                    break

                tries += 1
                stochastic = (np.random.rand() < cfg.stochastic_ratio)

                try:
                    build_vec, level_vec, piles_vec = self._decode_one_solution(
                        priority_logits=priority_logits,
                        count_pred=count_pred,
                        capacity_logits=capacity_logits,
                        valid_mask=valid_mask,
                        k_override=kk,
                        stochastic=stochastic,
                        cfg=cfg,
                    )
                except Exception:
                    continue

                sol = pack_solution_fn(build_vec, piles_vec, level_vec, context)
                added = try_add_solution(sol)
                if added:
                    made_progress = True

            if not made_progress:
                break

        if len(sols) == 0:
            return np.empty((0, dim), dtype=float)

        sols = sanitize_population(
            np.asarray(sols, dtype=float),
            dim=dim,
            lb_vec=lb_vec,
            ub_vec=ub_vec,
        )
        return sols


def make_pcc_seed_builder(
    ckpt_path: str,
    decode_cfg: Optional[SeedDecodeConfig] = None,
    device: Optional[str] = None,
) -> Callable[..., np.ndarray]:
    """
    返回一个 builder_fn，可直接给 MOGABKA 的 seeding 配置使用。
    context 里需要至少提供:
        - x_raw_flat
    可选:
        - pack_solution_fn
        - repair_fn
        - evaluate_fn
    """
    cfg = decode_cfg or SeedDecodeConfig()
    generator_box = {"obj": None}

    def builder_fn(
        num_needed: int,
        dim: int,
        lb_vec: np.ndarray,
        ub_vec: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        context = context or {}
        if "x_raw_flat" not in context:
            raise KeyError("context 中缺少 x_raw_flat")

        if int(num_needed) <= 0:
            return np.empty((0, dim), dtype=float)

        debug = bool(context.get("debug_seed_injection", False))

        # 懒加载：只有真正需要种子时才加载 checkpoint
        if generator_box["obj"] is None:
            generator_box["obj"] = PCCSeedGenerator(ckpt_path=ckpt_path, device=device)

        generator = generator_box["obj"]

        local_cfg = SeedDecodeConfig(
            num_seeds=max(int(num_needed), 1),
            count_radius=cfg.count_radius,
            top_margin=cfg.top_margin,
            stochastic_ratio=cfg.stochastic_ratio,
            priority_temperature=cfg.priority_temperature,
            capacity_temperature=cfg.capacity_temperature,
            min_k=cfg.min_k,
            max_k=cfg.max_k,
        )

        pack_solution_fn = context.get("pack_solution_fn", default_pack_solution)
        repair_fn = context.get("repair_fn", None)
        evaluate_fn = context.get("evaluate_fn", None)

        strict_solutions = generator.generate_packed_seed_solutions(
            x_raw_flat=context["x_raw_flat"],
            dim=dim,
            lb_vec=lb_vec,
            ub_vec=ub_vec,
            pack_solution_fn=pack_solution_fn,
            repair_fn=repair_fn,
            evaluate_fn=evaluate_fn,
            context=context,
            decode_cfg=local_cfg,
            keep_only_valid=True,
        )

        if debug:
            print(f"[seed_builder] strict valid seeds: {strict_solutions.shape[0]}/{int(num_needed)}")

        if strict_solutions.shape[0] >= int(num_needed) or evaluate_fn is None:
            return strict_solutions

        if not bool(context.get("allow_infeasible_seed_fallback", True)):
            return strict_solutions

        relaxed_solutions = generator.generate_packed_seed_solutions(
            x_raw_flat=context["x_raw_flat"],
            dim=dim,
            lb_vec=lb_vec,
            ub_vec=ub_vec,
            pack_solution_fn=pack_solution_fn,
            repair_fn=repair_fn,
            evaluate_fn=None,
            context=context,
            decode_cfg=local_cfg,
            keep_only_valid=False,
        )

        if debug:
            print(f"[seed_builder] relaxed structural seeds: {relaxed_solutions.shape[0]}/{int(num_needed)}")

        if strict_solutions.shape[0] == 0:
            return relaxed_solutions[: int(num_needed)]

        merged = np.vstack([strict_solutions, relaxed_solutions])
        merged = sanitize_population(
            merged,
            dim=dim,
            lb_vec=lb_vec,
            ub_vec=ub_vec,
        )
        return merged[: int(num_needed)]

    return builder_fn
