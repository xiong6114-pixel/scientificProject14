import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# =========================
# 常量
# =========================
N_POINTS = 100
FEAT_DIM = 49
IN_DIM = N_POINTS * FEAT_DIM
IGNORE_INDEX = -100


# =========================
# 工具函数
# =========================
def load_csv_float(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _fix_mat_array_shape(arr: np.ndarray, target_ndim: int) -> np.ndarray:
    """
    MATLAB v7.3 用 h5py 读出来后，二维数组常见情况是转置的。
    这里做一个尽量稳的修正：
    - 如果是 2D，就先转置一次再判断
    - 如果已经是对的形状，也保留
    """
    arr = np.array(arr)

    if arr.ndim != target_ndim:
        raise ValueError(f"期望 {target_ndim} 维数组，实际 {arr.ndim} 维")

    if arr.ndim == 2:
        # h5py 读 MATLAB 2D 常常是列优先存的，通常需要转置
        # 所以先做一次转置，后面再由上层逻辑检查是否合理
        arr_t = arr.T
        return arr_t

    return arr


def _read_mat_v73(path: str) -> Dict[str, np.ndarray]:
    """
    读取 MATLAB -v7.3 mat 文件
    """
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    return out


def load_labels_v2_from_mat(mat_path: str) -> Dict[str, np.ndarray]:
    """
    从 labels_v2.mat 中读取：
    - Y_priority:      [N,100]
    - Y_count:         [N,1] 或 [N]
    - Y_capacity:      [N,100]
    - Y_valid_mask:    [N,100]
    - Y_capacity_mask: [N,100]
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"找不到 {mat_path}")

    raw = _read_mat_v73(mat_path)

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

    Y_priority = _fix_mat_array_shape(raw["Y_priority"], target_ndim=2).astype(np.int64)
    Y_count = _fix_mat_array_shape(raw["Y_count"], target_ndim=2).astype(np.int64)
    Y_capacity = _fix_mat_array_shape(raw["Y_capacity"], target_ndim=2).astype(np.int64)
    Y_valid_mask = _fix_mat_array_shape(raw["Y_valid_mask"], target_ndim=2).astype(np.int64)
    Y_capacity_mask = _fix_mat_array_shape(raw["Y_capacity_mask"], target_ndim=2).astype(np.int64)

    # Y_count 变成 [N]
    if Y_count.shape[1] == 1:
        Y_count = Y_count[:, 0]
    elif Y_count.shape[0] == 1:
        Y_count = Y_count[0]
    else:
        raise ValueError(f"Y_count 应为 [N,1] 或 [1,N]，实际 {Y_count.shape}")

    labels = {
        "Y_priority": Y_priority,
        "Y_count": Y_count,
        "Y_capacity": Y_capacity,
        "Y_valid_mask": Y_valid_mask,
        "Y_capacity_mask": Y_capacity_mask,
    }
    return labels


def compute_pad_mask_from_raw_X(X_raw_flat: np.ndarray) -> np.ndarray:
    """
    用原始 X（标准化前）推断 padding 点。
    约定：padding 点的 49 维特征全 0。
    返回: pad_mask [N,100]，True 表示该点是 padding。
    """
    if X_raw_flat.ndim != 2 or X_raw_flat.shape[1] != IN_DIM:
        raise ValueError(f"X 应为 [N,{IN_DIM}]，实际 {X_raw_flat.shape}")

    X3 = X_raw_flat.reshape(-1, N_POINTS, FEAT_DIM)
    pad_mask = np.all(X3 == 0.0, axis=-1)
    return pad_mask


class StandardScaler:
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

    # count 与 priority 一致性
    cnt_from_priority = Y_priority.sum(axis=1)
    if not np.all(cnt_from_priority == Y_count):
        raise ValueError("Y_count 与 Y_priority.sum(axis=1) 不一致")

    # capacity_mask 与 capacity 一致性
    if not np.all((Y_capacity != IGNORE_INDEX) == (Y_capacity_mask == 1)):
        raise ValueError("Y_capacity 与 Y_capacity_mask 不一致")

    # valid mask 与原始 padding 一致性检查
    pad_mask = compute_pad_mask_from_raw_X(X_raw)
    valid_mask_from_x = (~pad_mask).astype(np.int64)
    if not np.array_equal(valid_mask_from_x, Y_valid_mask):
        raise ValueError("Y_valid_mask 与 X 推断出的 valid mask 不一致")


class BlocksDatasetV2(Dataset):
    """
    对应 MATLAB labels_v2.mat 的数据集

    返回：
    x               : [100,49]
    y_priority      : [100] float
    y_count         : [] long / scalar
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


@dataclass
class SplitDataV2:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    Yp_train: np.ndarray
    Yp_val: np.ndarray
    Yp_test: np.ndarray

    Yc_train: np.ndarray
    Yc_val: np.ndarray
    Yc_test: np.ndarray

    Ycap_train: np.ndarray
    Ycap_val: np.ndarray
    Ycap_test: np.ndarray

    valid_train: np.ndarray
    valid_val: np.ndarray
    valid_test: np.ndarray

    capmask_train: np.ndarray
    capmask_val: np.ndarray
    capmask_test: np.ndarray

    scaler: StandardScaler


def build_splits_v2(
    X_csv_path: str,
    labels_mat_path: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SplitDataV2:
    X_raw = load_csv_float(X_csv_path)
    labels = load_labels_v2_from_mat(labels_mat_path)
    validate_v2_labels(X_raw, labels)

    N = X_raw.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(train_ratio * N)
    n_val = int(val_ratio * N)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx])
    X_val = scaler.transform(X_raw[val_idx])
    X_test = scaler.transform(X_raw[test_idx])

    return SplitDataV2(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,

        Yp_train=labels["Y_priority"][train_idx],
        Yp_val=labels["Y_priority"][val_idx],
        Yp_test=labels["Y_priority"][test_idx],

        Yc_train=labels["Y_count"][train_idx],
        Yc_val=labels["Y_count"][val_idx],
        Yc_test=labels["Y_count"][test_idx],

        Ycap_train=labels["Y_capacity"][train_idx],
        Ycap_val=labels["Y_capacity"][val_idx],
        Ycap_test=labels["Y_capacity"][test_idx],

        valid_train=labels["Y_valid_mask"][train_idx],
        valid_val=labels["Y_valid_mask"][val_idx],
        valid_test=labels["Y_valid_mask"][test_idx],

        capmask_train=labels["Y_capacity_mask"][train_idx],
        capmask_val=labels["Y_capacity_mask"][val_idx],
        capmask_test=labels["Y_capacity_mask"][test_idx],

        scaler=scaler,
    )