import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


DEFAULT_LEVEL_TO_PILES = {
    0: 0,
    1: 2,
    2: 4,
    3: 7,
}


class SetTransformerPriorityCountCapacity(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        count_mode: str = "dual_head",
        count_ordinal_dim: int = 50,
        count_class_num_classes: int = 51,
        count_fusion_alpha: float = 0.5,
        capacity_num_classes: int = 4,
        use_priority_capacity_gating: bool = False,
        use_selected_mask_capacity_gating: bool = False,
    ):
        super().__init__()
        self.count_mode = str(count_mode)
        self.count_ordinal_dim = int(count_ordinal_dim)
        self.count_class_num_classes = int(count_class_num_classes)
        self.count_fusion_alpha = float(count_fusion_alpha)
        self.capacity_num_classes = int(capacity_num_classes)
        self.use_priority_capacity_gating = bool(use_priority_capacity_gating)
        self.use_selected_mask_capacity_gating = bool(use_selected_mask_capacity_gating)

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
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.priority_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        if self.count_mode == "dual_head":
            self.count_ordinal_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.count_ordinal_dim),
            )
            self.count_class_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.count_class_num_classes),
            )
        else:
            self.count_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self._legacy_count_head_out_dim()),
            )

        capacity_in_dim = d_model
        if self.use_priority_capacity_gating:
            capacity_in_dim += 1
        if self.use_selected_mask_capacity_gating:
            capacity_in_dim += 1
        self.capacity_head = nn.Sequential(
            nn.Linear(capacity_in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.capacity_num_classes),
        )

    def _legacy_count_head_out_dim(self) -> int:
        if self.count_mode in {"ordinal", "flat_classification"}:
            return self.count_ordinal_dim if self.count_mode == "ordinal" else self.count_class_num_classes
        return 1

    @staticmethod
    def masked_mean_pool(h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_f = valid_mask.float().unsqueeze(-1)
        h_sum = (h * valid_f).sum(dim=1)
        denom = valid_f.sum(dim=1).clamp_min(1.0)
        return h_sum / denom

    @staticmethod
    def _clamp_count(count_pred: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_count = valid_mask.sum(dim=1).float()
        count_pred = torch.clamp(count_pred, min=0.0)
        return torch.minimum(count_pred, valid_count)

    @staticmethod
    def build_selected_mask(
        priority_logits: torch.Tensor,
        count_pred: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        selected_mask = torch.zeros_like(valid_mask)
        valid_count = valid_mask.sum(dim=1).long()
        k_vec = torch.round(count_pred).long().clamp_min(0)
        k_vec = torch.minimum(k_vec, valid_count)

        for b in range(priority_logits.shape[0]):
            k = int(k_vec[b].item())
            if k <= 0:
                continue
            idx = torch.where(valid_mask[b])[0]
            if idx.numel() == 0:
                continue
            scores = priority_logits[b, idx]
            chosen_local = torch.topk(scores, k=min(k, idx.numel()), dim=0).indices
            chosen_idx = idx[chosen_local]
            selected_mask[b, chosen_idx] = True

        return selected_mask

    def _decode_ordinal_count(
        self,
        count_ordinal_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        count_pred = torch.sigmoid(count_ordinal_logits).sum(dim=-1)
        return self._clamp_count(count_pred, valid_mask)

    def _decode_class_count(
        self,
        count_class_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        count_prob = torch.softmax(count_class_logits, dim=-1)
        count_value = torch.arange(
            self.count_class_num_classes,
            device=count_class_logits.device,
            dtype=count_class_logits.dtype,
        )
        count_expectation = torch.sum(count_prob * count_value.unsqueeze(0), dim=-1)
        count_expectation = self._clamp_count(count_expectation, valid_mask)

        class_pred = count_class_logits.argmax(dim=-1).long()
        class_pred = torch.minimum(class_pred, valid_mask.sum(dim=1).long())
        return count_expectation, class_pred

    def _build_dual_count_outputs(
        self,
        pooled: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ordinal_logits = self.count_ordinal_head(pooled)
        class_logits = self.count_class_head(pooled)
        ordinal_pred = self._decode_ordinal_count(ordinal_logits, valid_mask)
        class_expectation, class_pred = self._decode_class_count(class_logits, valid_mask)
        fused_pred = self._clamp_count(
            self.count_fusion_alpha * ordinal_pred
            + (1.0 - self.count_fusion_alpha) * class_expectation,
            valid_mask,
        )
        return {
            "ordinal_logits": ordinal_logits,
            "class_logits": class_logits,
            "ordinal_pred": ordinal_pred,
            "class_expectation": class_expectation,
            "class_pred": class_pred,
            "fused_pred": fused_pred,
        }

    def _build_legacy_count_outputs(
        self,
        pooled: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        count_logits = self.count_head(pooled)

        if self.count_mode == "ordinal":
            ordinal_pred = self._decode_ordinal_count(count_logits, valid_mask)
            class_expectation = ordinal_pred
            class_pred = torch.round(ordinal_pred).long()
            fused_pred = ordinal_pred
        elif self.count_mode == "flat_classification":
            class_expectation, class_pred = self._decode_class_count(count_logits, valid_mask)
            ordinal_pred = class_expectation
            fused_pred = class_expectation
        else:
            regression_pred = self._clamp_count(torch.nn.functional.softplus(count_logits.squeeze(-1)), valid_mask)
            ordinal_pred = regression_pred
            class_expectation = regression_pred
            class_pred = torch.round(regression_pred).long()
            fused_pred = regression_pred

        return {
            "ordinal_logits": count_logits if self.count_mode == "ordinal" else None,
            "class_logits": count_logits if self.count_mode == "flat_classification" else None,
            "ordinal_pred": ordinal_pred,
            "class_expectation": class_expectation,
            "class_pred": class_pred,
            "fused_pred": fused_pred,
        }

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        selected_mask: Optional[torch.Tensor] = None,
    ):
        valid_mask = ~key_padding_mask

        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        priority_logits = self.priority_head(h).squeeze(-1)
        pooled = self.masked_mean_pool(h, valid_mask)
        if self.count_mode == "dual_head":
            count_outputs = self._build_dual_count_outputs(pooled, valid_mask)
        else:
            count_outputs = self._build_legacy_count_outputs(pooled, valid_mask)

        if selected_mask is None:
            selected_mask = self.build_selected_mask(priority_logits, count_outputs["fused_pred"], valid_mask)
        else:
            selected_mask = selected_mask.bool() & valid_mask

        capacity_features = [h]
        if self.use_priority_capacity_gating:
            priority_prob = torch.sigmoid(priority_logits).unsqueeze(-1)
            capacity_features.append(priority_prob)
        if self.use_selected_mask_capacity_gating:
            capacity_features.append(selected_mask.float().unsqueeze(-1))
        capacity_input = torch.cat(capacity_features, dim=-1)
        capacity_logits = self.capacity_head(capacity_input)

        return priority_logits, count_outputs, capacity_logits, selected_mask


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


def _sol_key(sol: np.ndarray) -> Tuple[float, ...]:
    sol = np.asarray(sol, dtype=float).reshape(-1)
    return tuple(np.round(sol, 8).tolist())


@dataclass
class SeedDecodeConfig:
    num_seeds: int = 10
    count_radius: int = 2
    top_margin: int = 4
    stochastic_ratio: float = 0.5
    priority_temperature: float = 1.0
    capacity_temperature: float = 1.0
    min_k: int = 1
    max_k: Optional[int] = None


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

        self.n_points = int(ckpt["n_points"])
        self.feat_dim = int(ckpt["feat_dim"])
        self.in_dim = int(ckpt.get("in_dim", self.n_points * self.feat_dim))

        arch = ckpt["arch"]
        if "count_mode" in arch:
            count_mode = str(arch["count_mode"])
            count_ordinal_dim = int(arch.get("count_ordinal_dim", self.n_points))
            count_class_num_classes = int(arch.get("count_class_num_classes", self.n_points + 1))
            count_fusion_alpha = float(arch.get("count_fusion_alpha", 0.5))
        elif int(arch.get("count_num_classes", 1)) > 1:
            count_mode = "flat_classification"
            count_ordinal_dim = self.n_points
            count_class_num_classes = int(arch["count_num_classes"])
            count_fusion_alpha = 0.5
        else:
            count_mode = "regression"
            count_ordinal_dim = self.n_points
            count_class_num_classes = self.n_points + 1
            count_fusion_alpha = 0.5

        self.model = SetTransformerPriorityCountCapacity(
            feat_dim=self.feat_dim,
            d_model=arch["d_model"],
            nhead=arch["nhead"],
            num_layers=arch["num_layers"],
            dim_feedforward=arch["ff_dim"],
            dropout=arch["dropout"],
            count_mode=count_mode,
            count_ordinal_dim=count_ordinal_dim,
            count_class_num_classes=count_class_num_classes,
            count_fusion_alpha=count_fusion_alpha,
            capacity_num_classes=int(arch.get("capacity_num_classes", 4)),
            use_priority_capacity_gating=bool(arch.get("use_priority_capacity_gating", False)),
            use_selected_mask_capacity_gating=bool(arch.get("use_selected_mask_capacity_gating", False)),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_std = np.asarray(ckpt["scaler_std"], dtype=np.float32)
        ckpt_level_to_piles = ckpt.get("level_to_piles", None)
        if level_to_piles is not None:
            self.level_to_piles = {int(k): int(v) for k, v in level_to_piles.items()}
        elif ckpt_level_to_piles is not None:
            self.level_to_piles = {int(k): int(v) for k, v in ckpt_level_to_piles.items()}
        else:
            self.level_to_piles = DEFAULT_LEVEL_TO_PILES.copy()

    def compute_pad_mask_from_raw_X_single(self, x_raw_flat: np.ndarray) -> np.ndarray:
        x_raw_flat = np.asarray(x_raw_flat, dtype=np.float32).reshape(-1)
        if x_raw_flat.shape != (self.in_dim,):
            raise ValueError(f"x_raw_flat 应为 [{self.in_dim}]，实际 {x_raw_flat.shape}")
        x3 = x_raw_flat.reshape(self.n_points, self.feat_dim)
        return np.all(x3 == 0.0, axis=-1)

    def _prepare_model_inputs(self, x_raw_flat: np.ndarray):
        x_raw_flat = np.asarray(x_raw_flat, dtype=np.float32).reshape(-1)
        pad_mask = self.compute_pad_mask_from_raw_X_single(x_raw_flat)
        valid_mask = ~pad_mask
        x_norm = (x_raw_flat[None, :] - self.scaler_mean) / self.scaler_std
        x_tensor = torch.from_numpy(x_norm).float().to(self.device).reshape(1, self.n_points, self.feat_dim)
        pad_tensor = torch.from_numpy(pad_mask[None, :]).bool().to(self.device)
        return x_tensor, pad_tensor, pad_mask, valid_mask

    def default_pack_solution(
        self,
        build_vec: np.ndarray,
        piles_vec: np.ndarray,
        level_vec: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        build_vec = np.asarray(build_vec, dtype=float).reshape(-1)
        piles_vec = np.asarray(piles_vec, dtype=float).reshape(-1)

        if build_vec.shape[0] != self.n_points or piles_vec.shape[0] != self.n_points:
            raise ValueError(f"default_pack_solution 要求长度都是 {self.n_points}")

        return np.concatenate([build_vec, piles_vec], axis=0)

    @torch.no_grad()
    def predict(self, x_raw_flat: np.ndarray, selected_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        x_tensor, pad_tensor, pad_mask, valid_mask = self._prepare_model_inputs(x_raw_flat)

        selected_tensor = None
        if selected_mask is not None:
            selected_tensor = torch.from_numpy(
                np.asarray(selected_mask, dtype=bool).reshape(1, self.n_points)
            ).to(self.device)

        priority_logits_t, count_outputs_t, capacity_logits_t, selected_mask_t = self.model(
            x_tensor,
            key_padding_mask=pad_tensor,
            selected_mask=selected_tensor,
        )

        priority_logits = priority_logits_t[0].cpu().numpy()
        priority_prob = safe_sigmoid_np(priority_logits)
        ordinal_logits_t = count_outputs_t.get("ordinal_logits", None)
        class_logits_t = count_outputs_t.get("class_logits", None)
        ordinal_logits = None if ordinal_logits_t is None else np.asarray(ordinal_logits_t[0].cpu().numpy())
        class_logits = None if class_logits_t is None else np.asarray(class_logits_t[0].cpu().numpy())
        count_ordinal_pred = float(count_outputs_t["ordinal_pred"][0].cpu().item())
        count_class_expectation = float(count_outputs_t["class_expectation"][0].cpu().item())
        count_class_pred = int(count_outputs_t["class_pred"][0].cpu().item())
        count_pred = float(count_outputs_t["fused_pred"][0].cpu().item())
        capacity_logits = capacity_logits_t[0].cpu().numpy()
        selected_mask_used = np.asarray(selected_mask_t[0].cpu().numpy(), dtype=bool)

        return {
            "priority_logits": priority_logits,
            "priority_prob": priority_prob,
            "count_pred": count_pred,
            "count_expectation": count_pred,
            "count_ordinal_pred": count_ordinal_pred,
            "count_class_expectation": count_class_expectation,
            "count_class_pred": count_class_pred,
            "count_logits": class_logits if class_logits is not None else ordinal_logits,
            "count_ordinal_logits": ordinal_logits,
            "count_class_logits": class_logits,
            "capacity_logits": capacity_logits,
            "pad_mask": pad_mask,
            "valid_mask": valid_mask,
            "selected_mask": selected_mask_used,
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

    def _select_build_indices(
        self,
        priority_logits: np.ndarray,
        count_pred: float,
        valid_mask: np.ndarray,
        k_override: Optional[int],
        stochastic: bool,
        cfg: SeedDecodeConfig,
    ) -> np.ndarray:
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        if n_valid == 0:
            raise ValueError("没有有效候选点")

        base_k = round_and_clip_count(count_pred, n_valid)
        k = base_k if k_override is None else int(k_override)
        k = max(cfg.min_k, k)
        max_k = n_valid if cfg.max_k is None else int(cfg.max_k)
        k = min(max_k, k)
        k = min(k, n_valid)

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

        return np.asarray(chosen, dtype=np.int64)

    def _decode_capacity_for_chosen(
        self,
        capacity_logits: np.ndarray,
        chosen: np.ndarray,
        stochastic: bool,
        cfg: SeedDecodeConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        build_vec = np.zeros(self.n_points, dtype=np.int64)
        build_vec[chosen] = 1

        level_vec = np.zeros(self.n_points, dtype=np.int64)
        piles_vec = np.zeros(self.n_points, dtype=np.int64)

        for j in chosen:
            logits_j = capacity_logits[j]
            if logits_j.shape[0] >= 4:
                logits_123 = logits_j[1:4]
                level_ids = np.array([1, 2, 3], dtype=np.int64)
            else:
                logits_123 = logits_j
                level_ids = np.arange(1, logits_123.shape[0] + 1, dtype=np.int64)
            if stochastic:
                probs_123 = softmax_np(logits_123, temperature=cfg.capacity_temperature)
                lvl = int(np.random.choice(level_ids, p=probs_123))
            else:
                lvl = int(level_ids[int(np.argmax(logits_123))])

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
        pack_solution_fn = pack_solution_fn or self.default_pack_solution

        if int(cfg.num_seeds) <= 0:
            return np.empty((0, dim), dtype=float)

        pred = self.predict(x_raw_flat)
        priority_logits = pred["priority_logits"]
        count_pred = pred["count_pred"]
        valid_mask = pred["valid_mask"]

        n_valid = int(valid_mask.sum())
        if n_valid <= 0:
            return np.empty((0, dim), dtype=float)

        base_k = round_and_clip_count(count_pred, n_valid)
        if base_k <= 0:
            base_k = min(max(cfg.min_k, 1), n_valid)

        max_k = n_valid if cfg.max_k is None else int(cfg.max_k)
        k_list = []
        for dk in range(-cfg.count_radius, cfg.count_radius + 1):
            kk = base_k + dk
            kk = max(cfg.min_k, kk)
            kk = min(max_k, kk)
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

        try:
            chosen = self._select_build_indices(
                priority_logits=priority_logits,
                count_pred=count_pred,
                valid_mask=valid_mask,
                k_override=base_k,
                stochastic=False,
                cfg=cfg,
            )
            capacity_logits = self.predict(x_raw_flat, selected_mask=np.isin(np.arange(self.n_points), chosen))["capacity_logits"]
            build_vec, level_vec, piles_vec = self._decode_capacity_for_chosen(
                capacity_logits=capacity_logits,
                chosen=chosen,
                stochastic=False,
                cfg=cfg,
            )
            sol = pack_solution_fn(build_vec, piles_vec, level_vec, context)
            try_add_solution(sol)
        except Exception:
            pass

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
                    chosen = self._select_build_indices(
                        priority_logits=priority_logits,
                        count_pred=count_pred,
                        valid_mask=valid_mask,
                        k_override=kk,
                        stochastic=stochastic,
                        cfg=cfg,
                    )
                    capacity_logits = self.predict(
                        x_raw_flat,
                        selected_mask=np.isin(np.arange(self.n_points), chosen),
                    )["capacity_logits"]
                    build_vec, level_vec, piles_vec = self._decode_capacity_for_chosen(
                        capacity_logits=capacity_logits,
                        chosen=chosen,
                        stochastic=stochastic,
                        cfg=cfg,
                    )
                except Exception:
                    continue

                sol = pack_solution_fn(build_vec, piles_vec, level_vec, context)
                if try_add_solution(sol):
                    made_progress = True

            if not made_progress:
                break

        if len(sols) == 0:
            return np.empty((0, dim), dtype=float)

        return sanitize_population(np.asarray(sols, dtype=float), dim=dim, lb_vec=lb_vec, ub_vec=ub_vec)


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


def make_pcc_seed_builder(
    ckpt_path: str,
    decode_cfg: Optional[SeedDecodeConfig] = None,
    device: Optional[str] = None,
) -> Callable[..., np.ndarray]:
    generator = PCCSeedGenerator(ckpt_path=ckpt_path, device=device)
    cfg = decode_cfg or SeedDecodeConfig(max_k=generator.n_points)

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

        local_cfg = SeedDecodeConfig(
            num_seeds=max(int(num_needed), 1),
            count_radius=cfg.count_radius,
            top_margin=cfg.top_margin,
            stochastic_ratio=cfg.stochastic_ratio,
            priority_temperature=cfg.priority_temperature,
            capacity_temperature=cfg.capacity_temperature,
            min_k=cfg.min_k,
            max_k=generator.n_points if cfg.max_k is None else cfg.max_k,
        )

        pack_solution_fn = context.get("pack_solution_fn", generator.default_pack_solution)
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

        debug = bool(context.get("debug_seed_injection", False))
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
