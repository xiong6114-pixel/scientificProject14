import os
import random
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import scipy.io as sio
except Exception:
    sio = None


# =========================
# 配置
# =========================
X_PATH = "X_gridblocks_flat.csv"
LABELS_MAT_PATH = "labels_v2.mat"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-4

N_POINTS = 100
FEAT_DIM = 49
IN_DIM = N_POINTS * FEAT_DIM
IGNORE_INDEX = -100

# 模型超参数
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.10

# 多任务 loss 权重
W_PRIORITY = 1.0
W_COUNT = 0.3
W_CAPACITY = 0.7


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
    用原始 X（标准化前）推断 padding 点。
    约定：padding 点的 49 维特征全 0。
    返回: pad_mask [N,100]，True 表示该点是 padding。
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
    """
    修正 MATLAB 2D 数组读入后的形状。
    若 expected_cols 提供，则尽量变成 [N, expected_cols]。
    """
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
    """
    将 Y_count 修正为 [N]
    """
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


def load_labels_v2_from_mat(mat_path: str) -> Dict[str, np.ndarray]:
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

    return {
        "Y_priority": Y_priority,
        "Y_count": Y_count,
        "Y_capacity": Y_capacity,
        "Y_valid_mask": Y_valid_mask,
        "Y_capacity_mask": Y_capacity_mask,
    }


def validate_v2_labels(X_raw: np.ndarray, labels: Dict[str, np.ndarray]):
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


# =========================
# Dataset
# =========================
class BlocksDatasetV2(Dataset):
    """
    返回：
    x               : [100,49]
    y_priority      : [100] float
    y_count         : [] long
    y_capacity      : [100] long
    valid_mask      : [100] bool
    capacity_mask   : [100] bool
    pad_mask        : [100] bool
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
    """
    输入:
        x: [B,100,49]
        key_padding_mask: [B,100], True=padding

    输出:
        priority_logits: [B,100]
        count_pred:      [B]
        capacity_logits: [B,100,4]
    """
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
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

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

        self._init_bias()

    def _init_bias(self):
        last_p = self.priority_head[-1]
        last_c = self.count_head[-1]
        last_cap = self.capacity_head[-1]

        if isinstance(last_p, nn.Linear):
            nn.init.zeros_(last_p.bias)
        if isinstance(last_c, nn.Linear):
            nn.init.zeros_(last_c.bias)
        if isinstance(last_cap, nn.Linear):
            nn.init.zeros_(last_cap.bias)

    @staticmethod
    def masked_mean_pool(h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_f = valid_mask.float().unsqueeze(-1)   # [B,100,1]
        h_sum = (h * valid_f).sum(dim=1)             # [B,d]
        denom = valid_f.sum(dim=1).clamp_min(1.0)    # [B,1]
        pooled = h_sum / denom
        return pooled

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        valid_mask = ~key_padding_mask

        h = self.input_proj(x)                                    # [B,100,d]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        priority_logits = self.priority_head(h).squeeze(-1)       # [B,100]

        pooled = self.masked_mean_pool(h, valid_mask)             # [B,d]
        count_raw = self.count_head(pooled).squeeze(-1)           # [B]
        count_pred = torch.nn.functional.softplus(count_raw)      # 非负更稳

        capacity_logits = self.capacity_head(h)                   # [B,100,4]

        return priority_logits, count_pred, capacity_logits


# =========================
# loss / metrics
# =========================
def priority_bce_loss(
    priority_logits: torch.Tensor,
    y_priority: torch.Tensor,
    valid_mask: torch.Tensor,
    bce: nn.BCEWithLogitsLoss,
) -> torch.Tensor:
    logits_v = priority_logits[valid_mask]
    target_v = y_priority[valid_mask]

    if logits_v.numel() == 0:
        return priority_logits.sum() * 0.0

    return bce(logits_v, target_v)


def count_mse_loss(
    count_pred: torch.Tensor,
    y_count: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.mse_loss(count_pred.float(), y_count.float())


def capacity_ce_loss(
    capacity_logits: torch.Tensor,
    y_capacity: torch.Tensor,
    ce: nn.CrossEntropyLoss,
) -> torch.Tensor:
    B, P, C = capacity_logits.shape
    return ce(capacity_logits.reshape(B * P, C), y_capacity.reshape(B * P))


@torch.no_grad()
def compute_count_pred_rounded(
    count_pred: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    将 count 预测四舍五入，并裁剪到 [0, valid_points]
    """
    valid_counts = valid_mask.sum(dim=1).long()
    pred_round = torch.round(count_pred).long()
    pred_round = torch.clamp(pred_round, min=0)
    pred_round = torch.minimum(pred_round, valid_counts)
    return pred_round


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    bce: nn.BCEWithLogitsLoss,
    ce: nn.CrossEntropyLoss,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    n_batches = 0

    pri_correct = 0.0
    pri_total = 0.0

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

        priority_logits, count_pred, capacity_logits = model(X, key_padding_mask=pad_mask)

        loss_pri = priority_bce_loss(priority_logits, Yp, valid_mask, bce)
        loss_cnt = count_mse_loss(count_pred, Yc)
        loss_cap = capacity_ce_loss(capacity_logits, Ycap, ce)

        loss = W_PRIORITY * loss_pri + W_COUNT * loss_cnt + W_CAPACITY * loss_cap

        loss_sum += loss.item()
        n_batches += 1

        # priority acc
        pri_pred = (torch.sigmoid(priority_logits[valid_mask]) > 0.5).float()
        pri_true = Yp[valid_mask]
        pri_correct += (pri_pred == pri_true).float().sum().item()
        pri_total += pri_true.numel()

        # count metrics
        count_abs_err += (count_pred.float() - Yc.float()).abs().sum().item()
        count_total += Yc.numel()

        count_pred_round = compute_count_pred_rounded(count_pred, valid_mask)
        count_exact += (count_pred_round == Yc.long()).float().sum().item()

        # capacity acc（只统计 ignore 之外的位置）
        cap_valid = (Ycap != IGNORE_INDEX)
        if cap_valid.any():
            cap_pred = capacity_logits.argmax(dim=-1)
            cap_correct += (cap_pred[cap_valid] == Ycap[cap_valid]).float().sum().item()
            cap_total += cap_valid.sum().item()

    metrics = {
        "loss": loss_sum / max(n_batches, 1),
        "priority_acc": pri_correct / max(pri_total, 1.0),
        "count_mae": count_abs_err / max(count_total, 1.0),
        "count_acc": count_exact / max(count_total, 1.0),
        "capacity_acc": cap_correct / max(cap_total, 1.0),
    }
    return metrics


# =========================
# 主程序
# =========================
def main():
    set_seed(SEED)

    assert os.path.isfile(X_PATH), f"找不到 {X_PATH}"
    assert os.path.isfile(LABELS_MAT_PATH), f"找不到 {LABELS_MAT_PATH}"

    X_raw = load_csv_float(X_PATH)   # [N,4900]
    labels = load_labels_v2_from_mat(LABELS_MAT_PATH)
    validate_v2_labels(X_raw, labels)

    Y_priority = labels["Y_priority"]
    Y_count = labels["Y_count"]
    Y_capacity = labels["Y_capacity"]
    Y_valid_mask = labels["Y_valid_mask"]
    Y_capacity_mask = labels["Y_capacity_mask"]

    N = X_raw.shape[0]
    print(f"Loaded: X={X_raw.shape}")
    print(f"Y_priority={Y_priority.shape}, Y_count={Y_count.shape}, Y_capacity={Y_capacity.shape}")
    print(f"Y_valid_mask={Y_valid_mask.shape}, Y_capacity_mask={Y_capacity_mask.shape}")

    # ========= 数据切分 =========
    idx = np.arange(N)
    np.random.shuffle(idx)

    n_train = int(0.7 * N)
    n_val = int(0.15 * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # ========= 标准化（只用训练集拟合） =========
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx])
    X_val = scaler.transform(X_raw[val_idx])
    X_test = scaler.transform(X_raw[test_idx])

    Yp_train, Yp_val, Yp_test = Y_priority[train_idx], Y_priority[val_idx], Y_priority[test_idx]
    Yc_train, Yc_val, Yc_test = Y_count[train_idx], Y_count[val_idx], Y_count[test_idx]
    Ycap_train, Ycap_val, Ycap_test = Y_capacity[train_idx], Y_capacity[val_idx], Y_capacity[test_idx]
    valid_train, valid_val, valid_test = Y_valid_mask[train_idx], Y_valid_mask[val_idx], Y_valid_mask[test_idx]
    capmask_train, capmask_val, capmask_test = Y_capacity_mask[train_idx], Y_capacity_mask[val_idx], Y_capacity_mask[test_idx]

    # ========= priority 正类权重（只统计非 padding 点） =========
    yp_valid_train = Yp_train[valid_train == 1]
    pos = float(yp_valid_train.sum())
    neg = float(yp_valid_train.size - pos)
    priority_pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    print(f"priority_pos_weight = {priority_pos_weight.item():.4f}")

    # ========= capacity 类别权重（只统计真实建站位置） =========
    cap_valid_train = Ycap_train[Ycap_train != IGNORE_INDEX]
    if cap_valid_train.size == 0:
        raise ValueError("训练集没有可用于 capacity 的监督样本")

    cap_counts = np.bincount(cap_valid_train, minlength=4).astype(np.float32)
    cap_weight = np.zeros(4, dtype=np.float32)

    # class 0 不参与有效监督
    cap_weight[0] = 0.0
    cap_weight[1:] = 1.0 / (cap_counts[1:] + 1.0)
    cap_weight_sum = cap_weight[1:].sum()
    if cap_weight_sum > 0:
        cap_weight[1:] = cap_weight[1:] / cap_weight_sum * 3.0

    cap_weight_t = torch.tensor(cap_weight, dtype=torch.float32, device=DEVICE)

    print(f"capacity_class_counts(valid only) = {cap_counts}")
    print(f"capacity_class_weight = {cap_weight_t.detach().cpu().numpy()}")

    # ========= DataLoader =========
    train_ds = BlocksDatasetV2(X_train, Yp_train, Yc_train, Ycap_train, valid_train, capmask_train)
    val_ds = BlocksDatasetV2(X_val, Yp_val, Yc_val, Ycap_val, valid_val, capmask_val)
    test_ds = BlocksDatasetV2(X_test, Yp_test, Yc_test, Ycap_test, valid_test, capmask_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ========= Model =========
    model = SetTransformerPriorityCountCapacity(
        feat_dim=FEAT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=FF_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 损失
    bce = nn.BCEWithLogitsLoss(pos_weight=priority_pos_weight)
    ce = nn.CrossEntropyLoss(weight=cap_weight_t, ignore_index=IGNORE_INDEX)

    best_val_score = -1.0
    best_state = None

    # ========= Train =========
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        for Xb, Yp_b, Yc_b, Ycap_b, valid_mask_b, cap_mask_b, pad_mask_b in train_loader:
            Xb = Xb.to(DEVICE)                         # [B,100,49]
            Yp_b = Yp_b.to(DEVICE)                    # [B,100]
            Yc_b = Yc_b.to(DEVICE)                    # [B]
            Ycap_b = Ycap_b.to(DEVICE)                # [B,100]
            valid_mask_b = valid_mask_b.to(DEVICE)    # [B,100]
            cap_mask_b = cap_mask_b.to(DEVICE)        # [B,100]
            pad_mask_b = pad_mask_b.to(DEVICE)        # [B,100]

            priority_logits, count_pred, capacity_logits = model(Xb, key_padding_mask=pad_mask_b)

            loss_pri = priority_bce_loss(priority_logits, Yp_b, valid_mask_b, bce)
            loss_cnt = count_mse_loss(count_pred, Yc_b)
            loss_cap = capacity_ce_loss(capacity_logits, Ycap_b, ce)

            loss = W_PRIORITY * loss_pri + W_COUNT * loss_cnt + W_CAPACITY * loss_cap

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        # ===== 训练集 / 验证集评估 =====
        train_metrics = eval_model(model, train_loader, bce, ce)
        val_metrics = eval_model(model, val_loader, bce, ce)

        # 综合分数：越大越好
        val_score = (
            val_metrics["priority_acc"] +
            val_metrics["count_acc"] +
            val_metrics["capacity_acc"]
        ) / 3.0

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_priority_acc={train_metrics['priority_acc']:.4f} | "
                f"train_count_mae={train_metrics['count_mae']:.4f} | "
                f"train_count_acc={train_metrics['count_acc']:.4f} | "
                f"train_capacity_acc={train_metrics['capacity_acc']:.4f} || "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_priority_acc={val_metrics['priority_acc']:.4f} | "
                f"val_count_mae={val_metrics['count_mae']:.4f} | "
                f"val_count_acc={val_metrics['count_acc']:.4f} | "
                f"val_capacity_acc={val_metrics['capacity_acc']:.4f}"
            )

    # ========= Test =========
    if best_state is not None:
        model.load_state_dict(best_state)

    best_train_metrics = eval_model(model, train_loader, bce, ce)
    test_metrics = eval_model(model, test_loader, bce, ce)

    print("\n==== BEST MODEL TRAIN ====")
    print(f"train_loss={best_train_metrics['loss']:.4f}")
    print(f"train_priority_acc={best_train_metrics['priority_acc']:.4f}")
    print(f"train_count_mae={best_train_metrics['count_mae']:.4f}")
    print(f"train_count_acc={best_train_metrics['count_acc']:.4f}")
    print(f"train_capacity_acc={best_train_metrics['capacity_acc']:.4f}")

    print("\n==== BEST MODEL TEST ====")
    print(f"test_loss={test_metrics['loss']:.4f}")
    print(f"test_priority_acc={test_metrics['priority_acc']:.4f}")
    print(f"test_count_mae={test_metrics['count_mae']:.4f}")
    print(f"test_count_acc={test_metrics['count_acc']:.4f}")
    print(f"test_capacity_acc={test_metrics['capacity_acc']:.4f}")

    # ========= 保存模型 =========
    ckpt = {
        "model": model.state_dict(),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "ignore_index": IGNORE_INDEX,
        "n_points": N_POINTS,
        "feat_dim": FEAT_DIM,
        "arch": {
            "name": "SetTransformerPriorityCountCapacity",
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
        },
        "train_config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "loss_weight_priority": W_PRIORITY,
            "loss_weight_count": W_COUNT,
            "loss_weight_capacity": W_CAPACITY,
        },
        "priority_pos_weight": priority_pos_weight.detach().cpu().numpy(),
        "capacity_class_weight": cap_weight_t.detach().cpu().numpy(),
        "best_val_score": best_val_score,
    }
    torch.save(ckpt, "set_transformer_pcc_ckpt.pt")
    print("Saved: set_transformer_pcc_ckpt.pt")


if __name__ == "__main__":
    main()