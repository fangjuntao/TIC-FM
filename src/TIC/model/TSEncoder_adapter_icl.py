from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import torch
from torch import nn, Tensor



def _extract_state_dict(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    """Best-effort extraction of a PyTorch state_dict from a checkpoint object."""
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            val = ckpt_obj.get(key)
            if isinstance(val, dict):
                return {str(k).replace("module.", ""): v for k, v in val.items()}
        # Some checkpoints are already a state-dict mapping.
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return {str(k).replace("module.", ""): v for k, v in ckpt_obj.items()}
    raise ValueError("Unsupported tsencoder checkpoint format; expected a dict or state_dict.")


def build_TSEncoder(
    *,
    tsencoder_checkpoint: str | Path | None,
    device: torch.device | str | None = None,
    hidden_dim: int = 512,
    seq_len: int = 512,
    num_patches: int = 32,
    use_fddm: bool = False,
    num_channels: int = 1,
    strict: bool = False,
) -> nn.Module:
    """Build a TSEncoder and optionally load a checkpoint.

    This repo uses the TSEncoder implementation under `TIC.model.TSEncoder_dev`.
    The returned module accepts input shaped `(B, C, L)` where `L == seq_len`.
    """

    dev = torch.device(device) if device is not None else torch.device("cpu")

    # Import lazily: TSEncoder_dev pulls in heavier deps (einops, huggingface_hub, etc.).
    from TIC.model.TSEncoder_dev.architecture.architecture import (
        TSEncoder8M,
        TSEncoder8MWithFDDM,
    )

    if use_fddm:
        model: nn.Module = TSEncoder8MWithFDDM(
            seq_len=int(seq_len),
            hidden_dim=int(hidden_dim),
            num_patches=int(num_patches),
            num_channels=int(num_channels),
            device=str(dev),
            pre_training=False,
        )
    else:
        model = TSEncoder8M(
            seq_len=int(seq_len),
            hidden_dim=int(hidden_dim),
            num_patches=int(num_patches),
            device=str(dev),
            pre_training=False,
        )

    ckpt_path = Path(tsencoder_checkpoint) if tsencoder_checkpoint is not None else None
    if ckpt_path is not None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"TSEncoder checkpoint not found: {ckpt_path}")
        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = _extract_state_dict(ckpt_obj)
        model.load_state_dict(state_dict, strict=bool(strict))

    model.to(dev)
    model.eval()
    return model


@torch.no_grad()
def encode_with_TSEncoder(
    model: nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode a batch of time series with a TSEncoder.

    - `x` supports `(N, L)`, `(N, 1, L)` or `(N, C, L)`.
    - returns `(N, D)` where `D == model.hidden_dim`.
    """

    if x.dim() == 2:
        x = x[:, None, :]
    if x.dim() != 3:
        raise ValueError(f"Expected x of shape (N,L) or (N,C,L); got {tuple(x.shape)}")

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    x = x.to(dev)

    outs: list[torch.Tensor] = []
    bs = max(1, int(batch_size))
    for i in range(0, x.shape[0], bs):
        outs.append(model(x[i : i + bs]))
    return torch.cat(outs, dim=0)


class TokenMLPAdapter(nn.Module):
    """Token-wise adapter mapping TSEncoder embedding dim -> ICL dim.

    Expects input shape (B, T, D_tsencoder) and returns (B, T, D_icl).
    """

    def __init__(
        self,
        tsencoder_dim: int,
        icl_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        tsencoder_dim = int(tsencoder_dim)
        icl_dim = int(icl_dim)
        hidden_dim = int(hidden_dim) if hidden_dim is not None else icl_dim

        self.tsencoder_dim = tsencoder_dim
        self.icl_dim = icl_dim

        self.net = nn.Sequential(
            nn.LayerNorm(tsencoder_dim) if use_layernorm else nn.Identity(),
            nn.Linear(tsencoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden_dim, icl_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Adapter expects (B, T, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.tsencoder_dim:
            raise ValueError(
                f"Adapter last-dim mismatch: expected {self.tsencoder_dim}, got {x.shape[-1]}"
            )
        return self.net(x)
