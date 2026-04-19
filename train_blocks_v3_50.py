import os
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import scipy.io as sio
except Exception:
    sio = None

# =========================
# 配置
# =========================
PROJECT_DIR = Path(__file__).resolve().parent
POI_KMEANS_DIR = PROJECT_DIR.parent / "poi kmeans"


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


X_PATH = os.environ.get("X_PATH", str(POI_KMEANS_DIR / "X_50_flat.csv"))
LABELS_MAT_PATH = os.environ.get("LABELS_MAT_PATH", str(POI_KMEANS_DIR / "labels_v3_50.mat"))
FALLBACK_LABELS_MAT_PATH = os.environ.get("FALLBACK_LABELS_MAT_PATH", str(POI_KMEANS_DIR / "labels_v3_50_balanced.mat"))
SPLIT_MAT_PATH = os.environ.get("SPLIT_MAT_PATH", str(POI_KMEANS_DIR / "split_idx_v3_50_grouped.mat"))
CKPT_PATH = os.environ.get("CKPT_PATH", "set_transformer_pcc_ckpt_50.pt")
SUMMARY_PATH = os.environ.get("SUMMARY_PATH", "train_summary_50.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = _get_env_int("SEED", 42)
BATCH_SIZE = _get_env_int("BATCH_SIZE", 64)
EPOCHS = _get_env_int("EPOCHS", 300)
LR = _get_env_float("LR", 1e-3)
WEIGHT_DECAY = _get_env_float("WEIGHT_DECAY", 1e-4)
EARLY_STOPPING_PATIENCE = _get_env_int("EARLY_STOPPING_PATIENCE", 40)
GRAD_CLIP_NORM = _get_env_float("GRAD_CLIP_NORM", 1.0)

N_POINTS = 50
FEAT_DIM = 49
IN_DIM = N_POINTS * FEAT_DIM
IGNORE_INDEX = -100
COUNT_MODE = os.environ.get("COUNT_MODE", "dual_head").strip().lower()
COUNT_ORDINAL_DIM = N_POINTS
COUNT_CLASS_NUM_CLASSES = N_POINTS + 1
CAPACITY_NUM_CLASSES = 3

# 模型超参数
D_MODEL = _get_env_int("D_MODEL", 128)
NHEAD = _get_env_int("NHEAD", 4)
NUM_LAYERS = _get_env_int("NUM_LAYERS", 2)
FF_DIM = _get_env_int("FF_DIM", 256)
DROPOUT = _get_env_float("DROPOUT", 0.10)
USE_PRIORITY_CAPACITY_GATING = _get_env_bool("USE_PRIORITY_CAPACITY_GATING", True)
USE_SELECTED_MASK_CAPACITY_GATING = _get_env_bool("USE_SELECTED_MASK_CAPACITY_GATING", True)
USE_BALANCED_SAMPLER = _get_env_bool("USE_BALANCED_SAMPLER", True)
BALANCED_SAMPLER_POWER = _get_env_float("BALANCED_SAMPLER_POWER", 0.5)

# 多任务 loss 权重
W_PRIORITY = _get_env_float("W_PRIORITY", 1.0)
W_COUNT_ORDINAL = _get_env_float("W_COUNT_ORDINAL", 0.20)
W_COUNT_CLASS = _get_env_float("W_COUNT_CLASS", 0.20)
W_CAPACITY = _get_env_float("W_CAPACITY", 0.70)
W_CONSISTENCY = _get_env_float("W_CONSISTENCY", 0.05)

PRIORITY_FOCAL_GAMMA = _get_env_float("PRIORITY_FOCAL_GAMMA", 0.0)
COUNT_LABEL_SMOOTHING = _get_env_float("COUNT_LABEL_SMOOTHING", 0.02)
COUNT_CLASS_LABEL_SMOOTHING = _get_env_float("COUNT_CLASS_LABEL_SMOOTHING", 0.02)
COUNT_ORDINAL_POS_WEIGHT_MAX = _get_env_float("COUNT_ORDINAL_POS_WEIGHT_MAX", 5.0)
COUNT_FUSION_ALPHA = _get_env_float("COUNT_FUSION_ALPHA", 0.50)
CAPACITY_FOCAL_GAMMA = _get_env_float("CAPACITY_FOCAL_GAMMA", 1.5)
CAPACITY_CLASS_BALANCE_BETA = _get_env_float("CAPACITY_CLASS_BALANCE_BETA", 0.999)

# =========================
# 工具函数
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv_float(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def compute_pad_mask_from_raw_X(X_raw_flat: np.ndarray) -> np.ndarray:
    """
    约定：padding 点的 49 维特征全 0。
    返回: pad_mask [N,50]，True 表示该点是 padding。
    """
    X3 = X_raw_flat.reshape(-1, N_POINTS, FEAT_DIM)
    pad_mask = np.all(X3 == 0.0, axis=-1)
    return pad_mask


class StandardScaler:
    """只用训练集拟合，避免泄露"""
    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std = np.maximum(self.std, self.eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler 尚未 fit")
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


def _read_mat_any(path: str) -> Dict[str, np.ndarray]:
    """
    优先按 MATLAB v7.3(h5py) 读，失败再尝试 scipy.io.loadmat。
    """
    try:
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.array(f[k])
        return out
    except Exception:
        if sio is None:
            raise
        raw = sio.loadmat(path)
        return {k: v for k, v in raw.items() if not k.startswith("__")}


def _fix_2d_shape(arr: np.ndarray, expected_cols: Optional[int] = None) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim != 2:
        raise ValueError(f"期望 2 维数组，实际 {arr.ndim} 维")

    if expected_cols is None:
        return arr

    if arr.shape[1] == expected_cols:
        return arr
    if arr.shape[0] == expected_cols:
        return arr.T

    raise ValueError(f"无法将数组修正到 [N,{expected_cols}]，实际 shape={arr.shape}")


def _fix_count_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr)

    if arr.ndim == 1:
        return arr.astype(np.int64)

    if arr.ndim != 2:
        raise ValueError(f"Y_count 期望 1/2 维，实际 {arr.ndim} 维")

    if arr.shape[1] == 1:
        return arr[:, 0].astype(np.int64)
    if arr.shape[0] == 1:
        return arr[0, :].astype(np.int64)

    raise ValueError(f"Y_count 应为 [N,1] 或 [1,N]，实际 {arr.shape}")


def _maybe_read_scalar(raw: Dict[str, np.ndarray], key: str) -> Optional[int]:
    if key not in raw:
        return None
    arr = np.array(raw[key]).reshape(-1)
    if arr.size == 0:
        return None
    return int(round(float(arr[0])))


def load_labels_from_mat(mat_path: str) -> Dict[str, Any]:
    raw = _read_mat_any(mat_path)

    required_keys = [
        "Y_priority",
        "Y_count",
        "Y_capacity",
        "Y_valid_mask",
        "Y_capacity_mask",
    ]
    for k in required_keys:
        if k not in raw:
            raise KeyError(f"{mat_path} 中缺少变量: {k}")

    Y_priority = _fix_2d_shape(raw["Y_priority"], expected_cols=N_POINTS).astype(np.int64)
    Y_capacity = _fix_2d_shape(raw["Y_capacity"], expected_cols=N_POINTS).astype(np.int64)
    Y_valid_mask = _fix_2d_shape(raw["Y_valid_mask"], expected_cols=N_POINTS).astype(np.int64)
    Y_capacity_mask = _fix_2d_shape(raw["Y_capacity_mask"], expected_cols=N_POINTS).astype(np.int64)
    Y_count = _fix_count_shape(raw["Y_count"])

    meta = {
        "rep1": _maybe_read_scalar(raw, "rep1"),
        "rep2": _maybe_read_scalar(raw, "rep2"),
        "rep3": _maybe_read_scalar(raw, "rep3"),
        "t1": _maybe_read_scalar(raw, "t1"),
        "t2": _maybe_read_scalar(raw, "t2"),
    }

    return {
        "Y_priority": Y_priority,
        "Y_count": Y_count,
        "Y_capacity": Y_capacity,
        "Y_valid_mask": Y_valid_mask,
        "Y_capacity_mask": Y_capacity_mask,
        "meta": meta,
    }


def validate_labels(X_raw: np.ndarray, labels: Dict[str, Any]):
    N = X_raw.shape[0]

    Y_priority = labels["Y_priority"]
    Y_count = labels["Y_count"]
    Y_capacity = labels["Y_capacity"]
    Y_valid_mask = labels["Y_valid_mask"]
    Y_capacity_mask = labels["Y_capacity_mask"]

    if Y_priority.shape != (N, N_POINTS):
        raise ValueError(f"Y_priority 应为 [{N},{N_POINTS}]，实际 {Y_priority.shape}")
    if Y_capacity.shape != (N, N_POINTS):
        raise ValueError(f"Y_capacity 应为 [{N},{N_POINTS}]，实际 {Y_capacity.shape}")
    if Y_valid_mask.shape != (N, N_POINTS):
        raise ValueError(f"Y_valid_mask 应为 [{N},{N_POINTS}]，实际 {Y_valid_mask.shape}")
    if Y_capacity_mask.shape != (N, N_POINTS):
        raise ValueError(f"Y_capacity_mask 应为 [{N},{N_POINTS}]，实际 {Y_capacity_mask.shape}")
    if Y_count.shape != (N,):
        raise ValueError(f"Y_count 应为 [{N}]，实际 {Y_count.shape}")

    if not np.all(np.isin(np.unique(Y_priority), [0, 1])):
        raise ValueError("Y_priority 只能包含 0/1")

    if not np.all(np.isin(np.unique(Y_valid_mask), [0, 1])):
        raise ValueError("Y_valid_mask 只能包含 0/1")

    if not np.all(np.isin(np.unique(Y_capacity_mask), [0, 1])):
        raise ValueError("Y_capacity_mask 只能包含 0/1")

    valid_capacity_values = np.unique(Y_capacity)
    if not np.all(np.isin(valid_capacity_values, [IGNORE_INDEX, 1, 2, 3])):
        raise ValueError(f"Y_capacity 非法取值: {valid_capacity_values}")

    cnt_from_priority = Y_priority.sum(axis=1)
    if not np.all(cnt_from_priority == Y_count):
        raise ValueError("Y_count 与 Y_priority.sum(axis=1) 不一致")

    if not np.all((Y_capacity != IGNORE_INDEX) == (Y_capacity_mask == 1)):
        raise ValueError("Y_capacity 与 Y_capacity_mask 不一致")

    pad_mask = compute_pad_mask_from_raw_X(X_raw)
    valid_mask_from_x = (~pad_mask).astype(np.int64)
    if not np.array_equal(valid_mask_from_x, Y_valid_mask):
        raise ValueError("Y_valid_mask 与 X 推断出的 valid mask 不一致")


def _fix_idx_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.int64)
    arr = arr.astype(np.int64)
    if arr.min() >= 1:
        arr = arr - 1
    return arr


def load_split_indices(path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not os.path.isfile(path):
        return None
    raw = _read_mat_any(path)
    candidate_sets = [
        ("train_idx", "val_idx", "test_idx"),
        ("idx_train", "idx_val", "idx_test"),
        ("trainInd", "valInd", "testInd"),
    ]
    for a, b, c in candidate_sets:
        if a in raw and b in raw and c in raw:
            return _fix_idx_shape(raw[a]), _fix_idx_shape(raw[b]), _fix_idx_shape(raw[c])
    return None


# =========================
# Dataset
# =========================
class BlocksDataset(Dataset):
    """
    返回：
    x               : [50,49]
    y_priority      : [50] float
    y_count         : [] long
    y_capacity      : [50] long
    valid_mask      : [50] bool
    capacity_mask   : [50] bool
    pad_mask        : [50] bool
    """
    def __init__(
        self,
        X: np.ndarray,
        Y_priority: np.ndarray,
        Y_count: np.ndarray,
        Y_capacity: np.ndarray,
        Y_valid_mask: np.ndarray,
        Y_capacity_mask: np.ndarray,
    ):
        self.X = torch.from_numpy(X).float()
        self.Y_priority = torch.from_numpy(Y_priority).float()
        self.Y_count = torch.from_numpy(Y_count).long()
        self.Y_capacity = torch.from_numpy(Y_capacity).long()
        self.valid_mask = torch.from_numpy(Y_valid_mask).bool()
        self.capacity_mask = torch.from_numpy(Y_capacity_mask).bool()
        self.pad_mask = ~self.valid_mask

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx].reshape(N_POINTS, FEAT_DIM)
        return (
            x,
            self.Y_priority[idx],
            self.Y_count[idx],
            self.Y_capacity[idx],
            self.valid_mask[idx],
            self.capacity_mask[idx],
            self.pad_mask[idx],
        )


# =========================
# 模型
# =========================
class SetTransformerPriorityCountCapacity(nn.Module):
    def __init__(
        self,
        feat_dim: int = FEAT_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        count_mode: str = "dual_head",
        count_ordinal_dim: int = N_POINTS,
        count_class_num_classes: int = N_POINTS + 1,
        count_fusion_alpha: float = 0.5,
        capacity_num_classes: int = 3,
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

        self._init_bias()

    def _legacy_count_head_out_dim(self) -> int:
        if self.count_mode in {"ordinal", "flat_classification"}:
            return self.count_ordinal_dim if self.count_mode == "ordinal" else self.count_class_num_classes
        return 1

    def _init_bias(self):
        heads = [self.priority_head, self.capacity_head]
        if self.count_mode == "dual_head":
            heads.extend([self.count_ordinal_head, self.count_class_head])
        else:
            heads.append(self.count_head)

        for head in heads:
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.bias)

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
            class_pred = compute_count_pred_rounded(ordinal_pred, valid_mask)
            fused_pred = ordinal_pred
        elif self.count_mode == "flat_classification":
            class_expectation, class_pred = self._decode_class_count(count_logits, valid_mask)
            ordinal_pred = class_expectation
            fused_pred = class_expectation
        else:
            regression_pred = self._clamp_count(torch.nn.functional.softplus(count_logits.squeeze(-1)), valid_mask)
            ordinal_pred = regression_pred
            class_expectation = regression_pred
            class_pred = compute_count_pred_rounded(regression_pred, valid_mask)
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


# =========================
# loss / metrics
# =========================
def priority_focal_loss(
    priority_logits: torch.Tensor,
    y_priority: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    logits_v = priority_logits[valid_mask]
    target_v = y_priority[valid_mask]
    if logits_v.numel() == 0:
        return priority_logits.sum() * 0.0

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_v,
        target_v,
        pos_weight=pos_weight,
        reduction="none",
    )
    if gamma <= 0:
        return bce.mean()

    prob = torch.sigmoid(logits_v)
    pt = torch.where(target_v > 0.5, prob, 1.0 - prob)
    focal_factor = (1.0 - pt).pow(gamma)
    return (focal_factor * bce).mean()


def build_count_ordinal_targets(
    y_count: torch.Tensor,
    n_thresholds: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    thresholds = torch.arange(n_thresholds, device=y_count.device).unsqueeze(0)
    target = (y_count.long().unsqueeze(1) > thresholds).float()
    if label_smoothing > 0:
        target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return target


def count_ordinal_loss(
    count_ordinal_logits: Optional[torch.Tensor],
    y_count: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    if count_ordinal_logits is None:
        return y_count.float().sum() * 0.0
    if count_ordinal_logits.ndim == 1:
        return nn.functional.smooth_l1_loss(count_ordinal_logits.float(), y_count.float(), beta=1.0)

    target = build_count_ordinal_targets(
        y_count=y_count,
        n_thresholds=count_ordinal_logits.shape[1],
        label_smoothing=COUNT_LABEL_SMOOTHING,
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        count_ordinal_logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    return loss.mean()


def count_class_loss(
    count_class_logits: Optional[torch.Tensor],
    y_count: torch.Tensor,
    class_weight: torch.Tensor,
) -> torch.Tensor:
    if count_class_logits is None:
        return y_count.float().sum() * 0.0
    return torch.nn.functional.cross_entropy(
        count_class_logits,
        y_count.long(),
        weight=class_weight,
        label_smoothing=COUNT_CLASS_LABEL_SMOOTHING,
    )


def capacity_cb_focal_loss(
    capacity_logits: torch.Tensor,
    y_capacity: torch.Tensor,
    class_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    valid = (y_capacity != IGNORE_INDEX)
    if valid.sum().item() == 0:
        return capacity_logits.sum() * 0.0

    logits_v = capacity_logits[valid]
    target_v = (y_capacity[valid] - 1).long()

    log_prob = torch.nn.functional.log_softmax(logits_v, dim=-1)
    log_pt = log_prob.gather(1, target_v.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    focal = (1.0 - pt).pow(gamma) if gamma > 0 else torch.ones_like(pt)
    alpha = class_weight[target_v]
    return (-alpha * focal * log_pt).mean()


@torch.no_grad()
def compute_count_pred_rounded(
    count_pred: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid_counts = valid_mask.sum(dim=1).long()
    pred_round = torch.round(count_pred).long()
    pred_round = torch.clamp(pred_round, min=0)
    pred_round = torch.minimum(pred_round, valid_counts)
    return pred_round


def priority_count_consistency_loss(
    priority_logits: torch.Tensor,
    y_count: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    soft_count = (torch.sigmoid(priority_logits) * valid_mask.float()).sum(dim=1)
    return nn.functional.smooth_l1_loss(soft_count, y_count.float(), beta=1.0)


@torch.no_grad()
def compute_priority_topk_acc(
    priority_logits: torch.Tensor,
    y_priority: torch.Tensor,
    y_count: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    batch_score = 0.0
    batch_total = 0

    for b in range(priority_logits.shape[0]):
        valid_idx = torch.where(valid_mask[b])[0]
        if valid_idx.numel() == 0:
            continue

        k = int(y_count[b].item())
        k = max(0, min(k, int(valid_idx.numel())))
        if k == 0:
            batch_score += 1.0
            batch_total += 1
            continue

        logits_b = priority_logits[b, valid_idx]
        target_b = y_priority[b, valid_idx]
        topk_idx = torch.topk(logits_b, k=k, dim=0).indices
        hits = target_b[topk_idx].float().sum().item()
        batch_score += hits / float(k)
        batch_total += 1

    if batch_total == 0:
        return 0.0
    return batch_score / float(batch_total)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    priority_pos_weight: torch.Tensor,
    count_ordinal_pos_weight: torch.Tensor,
    count_class_weight: torch.Tensor,
    capacity_class_weight: torch.Tensor,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    n_batches = 0

    pri_correct = 0.0
    pri_total = 0.0
    pri_tp = 0.0
    pri_fp = 0.0
    pri_fn = 0.0
    pri_topk_sum = 0.0
    pri_topk_total = 0

    count_abs_err = 0.0
    count_total = 0.0
    count_exact = 0.0

    cap_correct = 0.0
    cap_total = 0.0

    for X, Yp, Yc, Ycap, valid_mask, cap_mask, pad_mask in loader:
        X = X.to(DEVICE)
        Yp = Yp.to(DEVICE)
        Yc = Yc.to(DEVICE)
        Ycap = Ycap.to(DEVICE)
        valid_mask = valid_mask.to(DEVICE)
        cap_mask = cap_mask.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)

        priority_logits, count_outputs, capacity_logits, selected_mask = model(
            X,
            key_padding_mask=pad_mask,
        )

        loss_pri = priority_focal_loss(
            priority_logits,
            Yp,
            valid_mask,
            pos_weight=priority_pos_weight,
            gamma=PRIORITY_FOCAL_GAMMA,
        )
        loss_cnt_ordinal = count_ordinal_loss(count_outputs["ordinal_logits"], Yc, count_ordinal_pos_weight)
        loss_cnt_class = count_class_loss(count_outputs["class_logits"], Yc, count_class_weight)
        loss_cap = capacity_cb_focal_loss(
            capacity_logits,
            Ycap,
            capacity_class_weight,
            gamma=CAPACITY_FOCAL_GAMMA,
        )
        loss_consistency = priority_count_consistency_loss(priority_logits, Yc, valid_mask)
        loss = (
            W_PRIORITY * loss_pri
            + W_COUNT_ORDINAL * loss_cnt_ordinal
            + W_COUNT_CLASS * loss_cnt_class
            + W_CAPACITY * loss_cap
            + W_CONSISTENCY * loss_consistency
        )

        loss_sum += loss.item()
        n_batches += 1

        pri_pred = (torch.sigmoid(priority_logits[valid_mask]) > 0.5).float()
        pri_true = Yp[valid_mask]
        pri_correct += (pri_pred == pri_true).float().sum().item()
        pri_total += pri_true.numel()
        pri_tp += ((pri_pred == 1) & (pri_true == 1)).float().sum().item()
        pri_fp += ((pri_pred == 1) & (pri_true == 0)).float().sum().item()
        pri_fn += ((pri_pred == 0) & (pri_true == 1)).float().sum().item()
        pri_topk_sum += compute_priority_topk_acc(priority_logits, Yp, Yc, valid_mask)
        pri_topk_total += 1

        count_abs_err += (count_outputs["ordinal_pred"].float() - Yc.float()).abs().sum().item()
        count_total += Yc.numel()

        count_exact += (count_outputs["class_pred"] == Yc.long()).float().sum().item()

        cap_valid = (Ycap != IGNORE_INDEX)
        if cap_valid.any():
            cap_pred = capacity_logits.argmax(dim=-1) + 1
            cap_correct += (cap_pred[cap_valid] == Ycap[cap_valid]).float().sum().item()
            cap_total += cap_valid.sum().item()

    pri_precision = pri_tp / max(pri_tp + pri_fp, 1.0)
    pri_recall = pri_tp / max(pri_tp + pri_fn, 1.0)
    pri_f1 = 2.0 * pri_precision * pri_recall / max(pri_precision + pri_recall, 1e-12)

    metrics = {
        "loss": loss_sum / max(n_batches, 1),
        "priority_acc": pri_correct / max(pri_total, 1.0),
        "priority_f1": pri_f1,
        "priority_topk_acc": pri_topk_sum / max(pri_topk_total, 1),
        "count_mae": count_abs_err / max(count_total, 1.0),
        "count_acc": count_exact / max(count_total, 1.0),
        "capacity_acc": cap_correct / max(cap_total, 1.0),
    }
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    priority_pos_weight: torch.Tensor,
    count_ordinal_pos_weight: torch.Tensor,
    count_class_weight: torch.Tensor,
    capacity_class_weight: torch.Tensor,
) -> Dict[str, float]:
    model.train()
    train_loss_sum = 0.0
    n_batches = 0

    for X, Yp, Yc, Ycap, valid_mask, cap_mask, pad_mask in loader:
        X = X.to(DEVICE)
        Yp = Yp.to(DEVICE)
        Yc = Yc.to(DEVICE)
        Ycap = Ycap.to(DEVICE)
        valid_mask = valid_mask.to(DEVICE)
        cap_mask = cap_mask.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)

        opt.zero_grad(set_to_none=True)

        priority_logits, count_outputs, capacity_logits, selected_mask = model(
            X,
            key_padding_mask=pad_mask,
            selected_mask=cap_mask,
        )

        loss_pri = priority_focal_loss(
            priority_logits,
            Yp,
            valid_mask,
            pos_weight=priority_pos_weight,
            gamma=PRIORITY_FOCAL_GAMMA,
        )
        loss_cnt_ordinal = count_ordinal_loss(count_outputs["ordinal_logits"], Yc, count_ordinal_pos_weight)
        loss_cnt_class = count_class_loss(count_outputs["class_logits"], Yc, count_class_weight)
        loss_cap = capacity_cb_focal_loss(
            capacity_logits,
            Ycap,
            capacity_class_weight,
            gamma=CAPACITY_FOCAL_GAMMA,
        )
        loss_consistency = priority_count_consistency_loss(priority_logits, Yc, valid_mask)
        loss = (
            W_PRIORITY * loss_pri
            + W_COUNT_ORDINAL * loss_cnt_ordinal
            + W_COUNT_CLASS * loss_cnt_class
            + W_CAPACITY * loss_cap
            + W_CONSISTENCY * loss_consistency
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()

        train_loss_sum += loss.item()
        n_batches += 1

    return {"loss": train_loss_sum / max(n_batches, 1)}


def compute_class_balanced_weights(class_counts: np.ndarray, beta: float) -> np.ndarray:
    class_counts = np.asarray(class_counts, dtype=np.float32)
    weights = np.zeros_like(class_counts, dtype=np.float32)
    observed = class_counts > 0
    if not np.any(observed):
        return weights

    beta = min(max(float(beta), 0.0), 0.999999)
    effective_num = 1.0 - np.power(beta, class_counts[observed])
    weights_obs = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights_obs = weights_obs / np.mean(weights_obs)
    weights[observed] = weights_obs
    return weights


def compute_count_ordinal_pos_weight(y_count: np.ndarray, n_thresholds: int) -> np.ndarray:
    thresholds = np.arange(n_thresholds, dtype=np.int64)[None, :]
    positive = (y_count.reshape(-1, 1) > thresholds).sum(axis=0).astype(np.float32)
    total = float(y_count.shape[0])
    negative = total - positive
    pos_weight = np.ones((n_thresholds,), dtype=np.float32)
    observed = (positive > 0) & (negative > 0)
    pos_weight[observed] = np.sqrt(negative[observed] / np.maximum(positive[observed], 1.0))
    pos_weight = np.minimum(pos_weight, COUNT_ORDINAL_POS_WEIGHT_MAX)
    return pos_weight


def compute_count_class_weight(y_count: np.ndarray, n_classes: int) -> np.ndarray:
    count_class_counts = np.bincount(y_count.astype(np.int64), minlength=n_classes).astype(np.float32)
    class_weight = np.zeros((n_classes,), dtype=np.float32)
    observed = count_class_counts > 0
    class_weight[observed] = 1.0 / np.sqrt(count_class_counts[observed])
    if np.any(observed):
        class_weight[observed] /= class_weight[observed].mean()
    return class_weight


def initialize_output_biases(
    model: SetTransformerPriorityCountCapacity,
    y_priority: np.ndarray,
    y_count: np.ndarray,
    y_capacity: np.ndarray,
):
    def _logit(prob: np.ndarray) -> np.ndarray:
        prob = np.clip(np.asarray(prob, dtype=np.float32), 1e-4, 1.0 - 1e-4)
        return np.log(prob / (1.0 - prob))

    with torch.no_grad():
        last_priority = model.priority_head[-1]
        if isinstance(last_priority, nn.Linear):
            nn.init.zeros_(last_priority.weight)
            priority_prob = float(np.mean(y_priority > 0.5))
            last_priority.bias.fill_(float(_logit(np.array(priority_prob))))

        if model.count_mode == "dual_head":
            last_count_ordinal = model.count_ordinal_head[-1]
            if isinstance(last_count_ordinal, nn.Linear):
                nn.init.zeros_(last_count_ordinal.weight)
                thresholds = np.arange(model.count_ordinal_dim, dtype=np.int64)[None, :]
                count_prob = (y_count.reshape(-1, 1) > thresholds).mean(axis=0)
                last_count_ordinal.bias.copy_(
                    torch.from_numpy(_logit(count_prob)).to(last_count_ordinal.bias.device)
                )

            last_count_class = model.count_class_head[-1]
            if isinstance(last_count_class, nn.Linear):
                nn.init.zeros_(last_count_class.weight)
                count_hist = np.bincount(
                    y_count.astype(np.int64),
                    minlength=model.count_class_num_classes,
                ).astype(np.float32)
                count_prob = count_hist / np.maximum(np.sum(count_hist), 1.0)
                last_count_class.bias.copy_(
                    torch.from_numpy(np.log(np.clip(count_prob, 1e-6, None))).to(last_count_class.bias.device)
                )
        else:
            last_count = model.count_head[-1]
            if isinstance(last_count, nn.Linear):
                nn.init.zeros_(last_count.weight)
                if model.count_mode == "ordinal":
                    thresholds = np.arange(model.count_ordinal_dim, dtype=np.int64)[None, :]
                    count_prob = (y_count.reshape(-1, 1) > thresholds).mean(axis=0)
                    last_count.bias.copy_(torch.from_numpy(_logit(count_prob)).to(last_count.bias.device))
                elif model.count_mode == "flat_classification":
                    count_hist = np.bincount(
                        y_count.astype(np.int64),
                        minlength=model.count_class_num_classes,
                    ).astype(np.float32)
                    count_prob = count_hist / np.maximum(np.sum(count_hist), 1.0)
                    last_count.bias.copy_(torch.from_numpy(np.log(np.clip(count_prob, 1e-6, None))).to(last_count.bias.device))
                else:
                    last_count.bias.fill_(float(np.log(max(float(np.mean(y_count)), 1e-3))))

        last_capacity = model.capacity_head[-1]
        if isinstance(last_capacity, nn.Linear):
            nn.init.zeros_(last_capacity.weight)
            valid_capacity = y_capacity[y_capacity != IGNORE_INDEX]
            if valid_capacity.size > 0:
                cap_hist = np.bincount((valid_capacity - 1).astype(np.int64), minlength=model.capacity_num_classes).astype(np.float32)
                cap_prob = cap_hist / np.maximum(np.sum(cap_hist), 1.0)
                last_capacity.bias.copy_(torch.from_numpy(np.log(np.clip(cap_prob, 1e-6, None))).to(last_capacity.bias.device))


def choose_best_label_file() -> str:
    if os.path.isfile(LABELS_MAT_PATH):
        return LABELS_MAT_PATH
    if os.path.isfile(FALLBACK_LABELS_MAT_PATH):
        return FALLBACK_LABELS_MAT_PATH
    raise FileNotFoundError(f"找不到 {LABELS_MAT_PATH} 或 {FALLBACK_LABELS_MAT_PATH}")


def save_json(path: str, obj: Dict[str, Any]):
    def _default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        return str(x)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def main():
    set_seed(SEED)

    label_path = choose_best_label_file()

    assert os.path.isfile(X_PATH), f"找不到 {X_PATH}"
    assert os.path.isfile(label_path), f"找不到 {label_path}"

    X_raw = load_csv_float(X_PATH)
    labels = load_labels_from_mat(label_path)
    validate_labels(X_raw, labels)

    Y_priority = labels["Y_priority"]
    Y_count = labels["Y_count"]
    Y_capacity = labels["Y_capacity"]
    Y_valid_mask = labels["Y_valid_mask"]
    Y_capacity_mask = labels["Y_capacity_mask"]
    label_meta = labels.get("meta", {})

    N = X_raw.shape[0]
    print(f"Loaded: X={X_raw.shape}")
    print(f"Labels from: {label_path}")
    print(f"Y_priority={Y_priority.shape}, Y_count={Y_count.shape}, Y_capacity={Y_capacity.shape}")
    print(f"Y_valid_mask={Y_valid_mask.shape}, Y_capacity_mask={Y_capacity_mask.shape}")

    split_data = load_split_indices(SPLIT_MAT_PATH)
    if split_data is not None:
        train_idx, val_idx, test_idx = split_data
        print(f"Using grouped split: {SPLIT_MAT_PATH}")
    else:
        idx = np.arange(N)
        np.random.shuffle(idx)
        n_train = int(0.7 * N)
        n_val = int(0.15 * N)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        print("Using random split (grouped split file not found).")

    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx])
    X_val = scaler.transform(X_raw[val_idx])
    X_test = scaler.transform(X_raw[test_idx])

    Yp_train, Yp_val, Yp_test = Y_priority[train_idx], Y_priority[val_idx], Y_priority[test_idx]
    Yc_train, Yc_val, Yc_test = Y_count[train_idx], Y_count[val_idx], Y_count[test_idx]
    Ycap_train, Ycap_val, Ycap_test = Y_capacity[train_idx], Y_capacity[val_idx], Y_capacity[test_idx]
    valid_train, valid_val, valid_test = Y_valid_mask[train_idx], Y_valid_mask[val_idx], Y_valid_mask[test_idx]
    capmask_train, capmask_val, capmask_test = Y_capacity_mask[train_idx], Y_capacity_mask[val_idx], Y_capacity_mask[test_idx]

    yp_valid_train = Yp_train[valid_train == 1]
    pos = float(yp_valid_train.sum())
    neg = float(yp_valid_train.size - pos)
    priority_pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    print(f"priority_pos_weight = {priority_pos_weight.item():.4f}")

    cap_valid_train = Ycap_train[Ycap_train != IGNORE_INDEX]
    if cap_valid_train.size == 0:
        raise ValueError("训练集没有可用于 capacity 的监督样本")

    cap_counts = np.bincount((cap_valid_train - 1).astype(np.int64), minlength=CAPACITY_NUM_CLASSES).astype(np.float32)
    cap_weight = compute_class_balanced_weights(cap_counts, beta=CAPACITY_CLASS_BALANCE_BETA)
    cap_weight_t = torch.tensor(cap_weight, dtype=torch.float32, device=DEVICE)
    print(f"capacity_class_counts(selected only) = {cap_counts.astype(np.int64)}")
    print(f"capacity_cb_weight = {cap_weight_t.detach().cpu().numpy()}")

    count_class_counts = np.bincount(Yc_train.astype(np.int64), minlength=COUNT_CLASS_NUM_CLASSES).astype(np.float32)
    count_ordinal_pos_weight = compute_count_ordinal_pos_weight(
        Yc_train.astype(np.int64),
        n_thresholds=COUNT_ORDINAL_DIM,
    )
    count_class_weight = compute_count_class_weight(
        Yc_train.astype(np.int64),
        n_classes=COUNT_CLASS_NUM_CLASSES,
    )
    count_ordinal_weight_t = torch.tensor(count_ordinal_pos_weight, dtype=torch.float32, device=DEVICE)
    count_class_weight_t = torch.tensor(count_class_weight, dtype=torch.float32, device=DEVICE)
    print(f"count_class_counts = {count_class_counts.astype(np.int64)}")
    print(f"count_ordinal_pos_weight = {count_ordinal_weight_t.detach().cpu().numpy()}")
    print(f"count_class_weight = {count_class_weight_t.detach().cpu().numpy()}")

    train_ds = BlocksDataset(X_train, Yp_train, Yc_train, Ycap_train, valid_train, capmask_train)
    val_ds = BlocksDataset(X_val, Yp_val, Yc_val, Ycap_val, valid_val, capmask_val)
    test_ds = BlocksDataset(X_test, Yp_test, Yc_test, Ycap_test, valid_test, capmask_test)

    if USE_BALANCED_SAMPLER:
        train_sample_weight = np.power(
            1.0 / np.maximum(count_class_counts[Yc_train.astype(np.int64)], 1.0),
            BALANCED_SAMPLER_POWER,
        ).astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_sample_weight),
            num_samples=len(train_sample_weight),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
        print(f"Using BalancedSampler(power={BALANCED_SAMPLER_POWER:.2f})")
    else:
        sampler = None
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        print("Using shuffled train loader without BalancedSampler")

    train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SetTransformerPriorityCountCapacity(
        feat_dim=FEAT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=FF_DIM,
        dropout=DROPOUT,
        count_mode=COUNT_MODE,
        count_ordinal_dim=COUNT_ORDINAL_DIM,
        count_class_num_classes=COUNT_CLASS_NUM_CLASSES,
        count_fusion_alpha=COUNT_FUSION_ALPHA,
        capacity_num_classes=CAPACITY_NUM_CLASSES,
        use_priority_capacity_gating=USE_PRIORITY_CAPACITY_GATING,
        use_selected_mask_capacity_gating=USE_SELECTED_MASK_CAPACITY_GATING,
    ).to(DEVICE)
    initialize_output_biases(model, Yp_train, Yc_train, Ycap_train)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    best_train_metrics = None
    best_val_metrics = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            priority_pos_weight,
            count_ordinal_weight_t,
            count_class_weight_t,
            cap_weight_t,
        )
        train_metrics = eval_model(
            model,
            train_eval_loader,
            priority_pos_weight,
            count_ordinal_weight_t,
            count_class_weight_t,
            cap_weight_t,
        )
        val_metrics = eval_model(
            model,
            val_loader,
            priority_pos_weight,
            count_ordinal_weight_t,
            count_class_weight_t,
            cap_weight_t,
        )

        scheduler.step(val_metrics["loss"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_priority_acc={val_metrics['priority_acc']:.4f} | "
            f"val_priority_f1={val_metrics['priority_f1']:.4f} | "
            f"val_priority_topk_acc={val_metrics['priority_topk_acc']:.4f} | "
            f"val_count_mae={val_metrics['count_mae']:.4f} | "
            f"val_count_acc={val_metrics['count_acc']:.4f} | "
            f"val_capacity_acc={val_metrics['capacity_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
        else:
            no_improve += 1

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("训练未得到可保存模型")

    model.load_state_dict(best_state)
    test_metrics = eval_model(
        model,
        test_loader,
        priority_pos_weight,
        count_ordinal_weight_t,
        count_class_weight_t,
        cap_weight_t,
    )

    print("\n==== BEST MODEL SUMMARY ====")
    print(f"best_epoch={best_epoch}")
    print(f"best_val_loss={best_val_loss:.4f}")
    print(f"train_priority_acc={best_train_metrics['priority_acc']:.4f}")
    print(f"train_priority_f1={best_train_metrics['priority_f1']:.4f}")
    print(f"train_priority_topk_acc={best_train_metrics['priority_topk_acc']:.4f}")
    print(f"train_count_mae={best_train_metrics['count_mae']:.4f}")
    print(f"train_count_acc={best_train_metrics['count_acc']:.4f}")
    print(f"train_capacity_acc={best_train_metrics['capacity_acc']:.4f}")

    print("\n==== BEST MODEL TEST ====")
    print(f"test_loss={test_metrics['loss']:.4f}")
    print(f"test_priority_acc={test_metrics['priority_acc']:.4f}")
    print(f"test_priority_f1={test_metrics['priority_f1']:.4f}")
    print(f"test_priority_topk_acc={test_metrics['priority_topk_acc']:.4f}")
    print(f"test_count_mae={test_metrics['count_mae']:.4f}")
    print(f"test_count_acc={test_metrics['count_acc']:.4f}")
    print(f"test_capacity_acc={test_metrics['capacity_acc']:.4f}")

    rep1 = label_meta.get("rep1")
    rep2 = label_meta.get("rep2")
    rep3 = label_meta.get("rep3")
    if rep1 is None or rep2 is None or rep3 is None:
        rep1, rep2, rep3 = 2, 4, 7

    level_to_piles = {
        0: 0,
        1: int(rep1),
        2: int(rep2),
        3: int(rep3),
    }

    ckpt = {
        "model": model.state_dict(),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "ignore_index": IGNORE_INDEX,
        "n_points": N_POINTS,
        "feat_dim": FEAT_DIM,
        "in_dim": IN_DIM,
        "arch": {
            "name": "SetTransformerPriorityCountCapacity",
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
            "count_mode": COUNT_MODE,
            "count_ordinal_dim": COUNT_ORDINAL_DIM,
            "count_class_num_classes": COUNT_CLASS_NUM_CLASSES,
            "count_fusion_alpha": COUNT_FUSION_ALPHA,
            "capacity_num_classes": CAPACITY_NUM_CLASSES,
            "use_priority_capacity_gating": USE_PRIORITY_CAPACITY_GATING,
            "use_selected_mask_capacity_gating": USE_SELECTED_MASK_CAPACITY_GATING,
        },
        "train_config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "loss_weight_priority": W_PRIORITY,
            "loss_weight_count_ordinal": W_COUNT_ORDINAL,
            "loss_weight_count_class": W_COUNT_CLASS,
            "loss_weight_capacity": W_CAPACITY,
            "loss_weight_consistency": W_CONSISTENCY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "use_balanced_sampler": USE_BALANCED_SAMPLER,
            "balanced_sampler_power": BALANCED_SAMPLER_POWER,
            "priority_focal_gamma": PRIORITY_FOCAL_GAMMA,
            "count_label_smoothing": COUNT_LABEL_SMOOTHING,
            "count_class_label_smoothing": COUNT_CLASS_LABEL_SMOOTHING,
            "count_ordinal_pos_weight_max": COUNT_ORDINAL_POS_WEIGHT_MAX,
            "count_fusion_alpha": COUNT_FUSION_ALPHA,
            "capacity_focal_gamma": CAPACITY_FOCAL_GAMMA,
            "capacity_class_balance_beta": CAPACITY_CLASS_BALANCE_BETA,
        },
        "priority_pos_weight": priority_pos_weight.detach().cpu().numpy(),
        "count_ordinal_pos_weight": count_ordinal_weight_t.detach().cpu().numpy(),
        "count_class_weight": count_class_weight_t.detach().cpu().numpy(),
        "capacity_cb_weight": cap_weight_t.detach().cpu().numpy(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "label_path": label_path,
        "x_path": X_PATH,
        "split_path": SPLIT_MAT_PATH if os.path.isfile(SPLIT_MAT_PATH) else None,
        "label_meta": label_meta,
        "level_to_piles": level_to_piles,
    }

    torch.save(ckpt, CKPT_PATH)
    print(f"Saved: {CKPT_PATH}")

    summary = {
        "device": DEVICE,
        "x_path": X_PATH,
        "label_path": label_path,
        "split_path": SPLIT_MAT_PATH if os.path.isfile(SPLIT_MAT_PATH) else None,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "level_to_piles": level_to_piles,
        "label_meta": label_meta,
    }
    save_json(SUMMARY_PATH, summary)
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
