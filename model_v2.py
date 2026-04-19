import torch
import torch.nn as nn

N_POINTS = 100
FEAT_DIM = 49


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

        # 点级 priority
        self.priority_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # 全局 pooled 表示 -> count 回归
        self.count_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # 点级 capacity
        # 仍然输出 4 类：0/1/2/3
        # 训练时未建站位置会被 ignore_index=-100 跳过
        self.capacity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )

        self._init_bias()

    def _init_bias(self):
        # 让初始输出更平稳
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
        """
        h: [B,100,d]
        valid_mask: [B,100], True=真实点
        return: [B,d]
        """
        valid_f = valid_mask.float().unsqueeze(-1)   # [B,100,1]
        h_sum = (h * valid_f).sum(dim=1)             # [B,d]
        denom = valid_f.sum(dim=1).clamp_min(1.0)    # [B,1]
        pooled = h_sum / denom
        return pooled

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        """
        x: [B,100,49]
        key_padding_mask: [B,100], True=padding
        """
        valid_mask = ~key_padding_mask

        h = self.input_proj(x)                                    # [B,100,d]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        priority_logits = self.priority_head(h).squeeze(-1)       # [B,100]

        pooled = self.masked_mean_pool(h, valid_mask)             # [B,d]
        count_pred = self.count_head(pooled).squeeze(-1)          # [B]

        capacity_logits = self.capacity_head(h)                   # [B,100,4]

        return priority_logits, count_pred, capacity_logits