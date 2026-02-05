from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Ensure we import the local workspace package (repo_root/src)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from TIC.model.TSEncoder_adapter_icl import TokenMLPAdapter, build_TSEncoder  # noqa: E402
from TIC.prior.data_reader import DataReader  # noqa: E402
from TIC.sklearn.classifier import TSEncoderICLClassifierV2  # noqa: E402
from TIC.model.TSEncoder_dev.adapters import VarianceBasedSelector  # noqa: E402

from TIC.model.orion_learning import ICLearning  # noqa: E402


def _print_effective_runtime_config(*, title: str, payload: dict) -> None:
    """Pretty-print a JSON payload for reproducibility/debugging."""
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        # Fallback to best-effort string representation.
        text = str(payload)
    print(f"\n{title}\n{text}\n")


def _remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map labels to contiguous ints [0..K-1] based on train set."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    classes = np.unique(y_train)
    cls_to_id = {c: i for i, c in enumerate(classes.tolist())}

    y_train_m = np.vectorize(cls_to_id.get)(y_train)
    y_test_m = np.vectorize(cls_to_id.get)(y_test)

    if np.any(y_test_m == None):  # noqa: E711
        missing = set(np.unique(y_test)) - set(classes)
        raise ValueError(f"Test labels contain unseen classes: {sorted(missing)}")

    return y_train_m.astype(np.int64), y_test_m.astype(np.int64), classes


def _select_support_indices(y: np.ndarray, support_size: int, seed: int) -> np.ndarray:
    """Pick support indices ensuring all classes appear at least once."""
    rng = np.random.RandomState(int(seed))
    y = np.asarray(y)
    classes = np.unique(y)
    n_classes = int(classes.shape[0])
    support_size = int(support_size)
    if support_size < n_classes:
        support_size = n_classes

    chosen: list[int] = []
    remaining: list[int] = []

    for c in classes:
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        pick = int(rng.choice(idx))
        chosen.append(pick)
        remaining.extend([int(i) for i in idx if int(i) != pick])

    need = support_size - len(chosen)
    if need > 0:
        remaining_np = np.array(remaining, dtype=np.int64)
        if remaining_np.size > 0:
            extra = rng.choice(remaining_np, size=min(need, remaining_np.size), replace=False)
            chosen.extend([int(i) for i in extra])

    return np.array(chosen, dtype=np.int64)


def _ensure_2d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce X into (N, L). Supports UCR (N,L) and UEA (N,C,L)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        if X.shape[1] == 1:
            return X[:, 0, :]
        # NOTE: for UEA multivariate series, we do NOT average channels here.
        # Use `_uea_concat_and_sample()` to concat channels and sample to a fixed length.
        return X
    raise ValueError(f"Unexpected X shape: {X.shape}")


def _uea_concat_and_sample(
    X: np.ndarray,
    *,
    target_len: int,
    seed: int,
    mode: str = "center",
) -> np.ndarray:
    """Convert UEA (N,C,L) into (N,target_len) by channel-concat then crop/pad.

    Spec required by user:
    - For multichannel series, concatenate channels into a single long 1D series.
    - Then sample a length `target_len` window (default 512) and feed to TSEncoder.
    """

    X = np.asarray(X, dtype=np.float32)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")

    if X.ndim == 2:
        X_flat = X
    elif X.ndim == 3:
        N, C, L = X.shape
        X_flat = X.reshape(N, C * L)
    else:
        raise ValueError(f"Unexpected UEA X shape: {X.shape}")

    N, Ltot = X_flat.shape
    if Ltot == target_len:
        return X_flat

    if Ltot < target_len:
        pad = np.zeros((N, target_len - Ltot), dtype=np.float32)
        return np.concatenate([X_flat, pad], axis=1)

    # Ltot > target_len: crop
    if mode not in {"center", "random"}:
        raise ValueError(f"mode must be 'center' or 'random', got {mode}")
    if mode == "center":
        start = (Ltot - target_len) // 2
        return X_flat[:, start : start + target_len]

    rng = np.random.RandomState(int(seed))
    # Deterministic but per-sample different windows
    starts = rng.randint(0, Ltot - target_len + 1, size=N)
    out = np.empty((N, target_len), dtype=np.float32)
    for i, s in enumerate(starts.tolist()):
        out[i] = X_flat[i, s : s + target_len]
    return out


def _ensure_3d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce X into (N, C, L)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        return X[None, None, :]
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Unexpected X shape for 3d series: {X.shape}")


def _uea_crop_pad_per_channel(
    X: np.ndarray,
    *,
    target_len: int,
    seed: int,
    mode: str = "center",
) -> np.ndarray:
    """Crop/pad UEA to (N,C,target_len) without mixing channels.

    Used by uea_fusion='sum_embed': each channel is fed into TSEncoder, then channel embeddings are summed.
    """

    X3 = _ensure_3d_timeseries(X)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")

    N, C, Ltot = X3.shape
    if Ltot == target_len:
        return X3
    if Ltot < target_len:
        pad = np.zeros((N, C, target_len - Ltot), dtype=np.float32)
        return np.concatenate([X3, pad], axis=2)

    if mode not in {"center", "random"}:
        raise ValueError(f"mode must be 'center' or 'random', got {mode}")
    if mode == "center":
        start = (Ltot - target_len) // 2
        return X3[:, :, start : start + target_len]

    rng = np.random.RandomState(int(seed))
    starts = rng.randint(0, Ltot - target_len + 1, size=N)
    out = np.empty((N, C, target_len), dtype=np.float32)
    for i, s in enumerate(starts.tolist()):
        out[i] = X3[i, :, s : s + target_len]
    return out


def _maybe_select_channels_uea(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    *,
    enabled: bool,
    new_num_channels: int | None,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply VarianceBasedSelector on UEA multichannel time series.

    Fits selector on training split only, then transforms train/test.
    Expects input in (N, C, L) (or will be coerced by _ensure_3d_timeseries).
    """

    if not enabled:
        return X_train_raw, X_test_raw
    if new_num_channels is None:
        return X_train_raw, X_test_raw

    X_train_np = _ensure_3d_timeseries(X_train_raw)
    X_test_np = _ensure_3d_timeseries(X_test_raw)

    if X_train_np.ndim != 3:
        return X_train_raw, X_test_raw

    _, c_train, _ = X_train_np.shape
    if c_train <= 1:
        return X_train_np, X_test_np

    k = int(new_num_channels)
    k = max(1, min(k, c_train))
    if k == c_train:
        return X_train_np, X_test_np

    if dataset_name is not None:
        print(f"[VarSelector][UEA] {dataset_name}: channels {c_train} -> {k}")

    selector = VarianceBasedSelector(k)
    selector.fit(X_train_np)
    X_train_sel = selector.transform(X_train_np)
    X_test_sel = selector.transform(X_test_np)
    return X_train_sel, X_test_sel


def _load_model_hparams_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid model_hparams_json: {path}")
    return obj


def _extract_full_state_dict(ckpt_obj: object) -> dict:
    if isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise ValueError("Invalid full model checkpoint: expected dict with 'state_dict' or raw state_dict.")


def _build_icl_predictor_from_hparams(hparams: dict) -> nn.Module:
    orion_cfg = hparams.get("orion", {}).get("icl_predictor", {}).get("config", {})
    adapter_cfg = hparams.get("adapter", {})

    icl_dim = adapter_cfg.get("icl_dim")
    if icl_dim is None:
        raise ValueError("Missing adapter.icl_dim in model_hparams_json")

    num_blocks = orion_cfg.get("icl_num_blocks")
    nhead = orion_cfg.get("icl_nhead")
    ff_factor = orion_cfg.get("ff_factor", 2)
    dropout = orion_cfg.get("dropout", 0.0)
    max_classes = orion_cfg.get("max_classes", 10)
    norm_first = orion_cfg.get("norm_first", True)
    perc_num_latents = orion_cfg.get("perc_num_latents", 0)
    perc_layers = orion_cfg.get("perc_layers", 0)

    if num_blocks is None or nhead is None:
        raise ValueError("Missing icl_num_blocks or icl_nhead in model_hparams_json")

    return ICLearning(
        max_classes=int(max_classes),
        d_model=int(icl_dim),
        num_blocks=int(num_blocks),
        nhead=int(nhead),
        dim_feedforward=int(icl_dim) * int(ff_factor),
        dropout=float(dropout),
        activation="gelu",
        norm_first=bool(norm_first),
        perc_num_latents=int(perc_num_latents),
        perc_layers=int(perc_layers),
    )


def _collect_model_hparams(
    *,
    custom_model: "_TSEncoderAdapterPlusOrionICL",
    tsencoder_ckpt: str,
    orion_ckpt: str,
    tsencoder_dim: int,
    icl_dim: int,
    adapter_hidden_dim_val: int | None,
    adapter_dropout_val: float,
    adapter_no_ln_val: bool,
    orion_cfg: dict,
    perc_num_latents: int | None,
    embed_dim: int,
    row_num_cls: int,
) -> dict:
    tsencoder_model = custom_model.tsencoder_model
    adapter = custom_model.adapter
    icl_predictor = custom_model.icl_predictor

    # Adapter details
    adapter_net = getattr(adapter, "net", None)
    adapter_linears: list[dict] = []
    adapter_layers: list[str] = []
    if adapter_net is not None:
        for layer in adapter_net:
            adapter_layers.append(layer.__class__.__name__)
            if isinstance(layer, nn.Linear):
                adapter_linears.append(
                    {
                        "in_features": int(layer.in_features),
                        "out_features": int(layer.out_features),
                        "bias": bool(layer.bias is not None),
                    }
                )

    # TSEncoder details
    tokgen_unit = getattr(tsencoder_model, "tokgen_unit", None)
    vit_unit = getattr(tsencoder_model, "vit_unit", None)
    transformer = getattr(vit_unit, "transformer", None) if vit_unit is not None else None
    tsencoder_layers = []
    tsencoder_transformer_layers = None
    if transformer is not None and hasattr(transformer, "layers"):
        tsencoder_transformer_layers = int(len(transformer.layers))
    if vit_unit is not None:
        tsencoder_layers.append("ViTUnit")
    if tokgen_unit is not None:
        tsencoder_layers.append("TokenGeneratorUnit")

    tsencoder_tokgen_convs = None
    tsencoder_scalar_encoders = None
    if tokgen_unit is not None:
        if hasattr(tokgen_unit, "convs"):
            tsencoder_tokgen_convs = int(len(tokgen_unit.convs))
        if hasattr(tokgen_unit, "scalar_encoders"):
            tsencoder_scalar_encoders = int(len(tokgen_unit.scalar_encoders))

    # Orion ICL predictor details (only this module is used)
    tf_icl = getattr(icl_predictor, "tf_icl", None)
    tf_icl_blocks = None
    if tf_icl is not None and hasattr(tf_icl, "blocks"):
        tf_icl_blocks = int(len(tf_icl.blocks))
    memory = getattr(icl_predictor, "memory", None)
    mem_write_layers = None
    mem_read_layers = None
    if memory is not None:
        if hasattr(memory, "write_layers"):
            mem_write_layers = int(len(memory.write_layers))
        if hasattr(memory, "read_layers"):
            mem_read_layers = int(len(memory.read_layers))

    # Pull icl_predictor-specific hyperparams from config if available
    icl_cfg = {}
    if isinstance(orion_cfg, dict):
        for k in (
            "embed_dim",
            "icl_nhead",
            "icl_num_blocks",
            "ff_factor",
            "dropout",
            "max_classes",
            "norm_first",
            "perc_layers",
            "perc_num_latents",
        ):
            if k in orion_cfg:
                icl_cfg[k] = orion_cfg[k]

    return {
        "model": "_TSEncoderAdapterPlusOrionICL",
        "structure": {
            "repr": str(custom_model),
            "tsencoder_model": str(tsencoder_model),
            "adapter": str(adapter),
            "icl_predictor": str(icl_predictor),
        },
        "tsencoder": {
            "ckpt": str(tsencoder_ckpt),
            "hidden_dim": int(getattr(tsencoder_model, "hidden_dim", tsencoder_dim)),
            "seq_len": int(getattr(custom_model, "tsencoder_seq_len", 512)),
            "batch_size": int(getattr(custom_model, "tsencoder_batch_size", 64)),
            "tsencoder_dim": int(tsencoder_dim),
        },
        "adapter": {
            "tsencoder_dim": int(tsencoder_dim),
            "icl_dim": int(icl_dim),
            "hidden_dim": (None if adapter_hidden_dim_val is None else int(adapter_hidden_dim_val)),
            "dropout": float(adapter_dropout_val),
            "use_layernorm": bool(not bool(adapter_no_ln_val)),
            "mlp": {
                "num_layers": (None if adapter_net is None else int(len(adapter_net))),
                "layer_types": adapter_layers,
                "linear_layers": adapter_linears,
            },
        },
        "orion": {
            "ckpt": str(orion_ckpt),
            "icl_predictor": {
                "config": icl_cfg,
                "tf_icl_blocks": tf_icl_blocks,
                "decoder": str(getattr(icl_predictor, "decoder", None)),
                "y_encoder": str(getattr(icl_predictor, "y_encoder", None)),
                "memory": {
                    "write_layers": mem_write_layers,
                    "read_layers": mem_read_layers,
                    "perc_num_latents": (None if perc_num_latents is None else int(perc_num_latents)),
                },
            },
        },
        "tsencoder_details": {
            "modules": tsencoder_layers,
            "transformer_layers": tsencoder_transformer_layers,
            "tokgen_convs": tsencoder_tokgen_convs,
            "tokgen_scalar_encoder_groups": tsencoder_scalar_encoders,
        },
    }


class _TSEncoderAdapterPlusOrionICL(nn.Module):
    """TSEncoder -> Adapter -> OrionMSP icl_predictor.

    Implements the forward signature expected by TSEncoderICLClassifierV2.
    """

    def __init__(
        self,
        *,
        tsencoder_model: nn.Module,
        adapter: nn.Module,
        icl_predictor: nn.Module,
        tsencoder_seq_len: int = 512,
        tsencoder_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.tsencoder_model = tsencoder_model
        self.adapter = adapter
        self.icl_predictor = icl_predictor
        self.tsencoder_seq_len = int(tsencoder_seq_len)
        self.tsencoder_batch_size = int(tsencoder_batch_size)

        self.max_classes = int(getattr(self.icl_predictor, "max_classes", 10))

    def train(self, mode: bool = True):
        super().train(mode)
        # evaluation-only: keep everything in eval
        self.tsencoder_model.eval()
        self.adapter.eval()
        self.icl_predictor.eval()
        return self

    def _pad_or_truncate(self, X: torch.Tensor) -> torch.Tensor:
        """Pad/truncate last dimension to tsencoder_seq_len.

        Supports:
        - Univariate: (B, T, L)
        - Multivariate: (B, T, C, L)
        """
        target = self.tsencoder_seq_len
        if X.shape[-1] == target:
            return X
        if X.shape[-1] > target:
            return X[..., :target]
        pad = X.new_zeros((*X.shape[:-1], target - X.shape[-1]))
        return torch.cat([X, pad], dim=-1)

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        """Encode rows with TSEncoder.

        - If X is (B,T,L): encodes each row.
        - If X is (B,T,C,L): encodes each channel separately then sums embeddings over C.
        """
        X = self._pad_or_truncate(X)

        device = next(self.tsencoder_model.parameters()).device
        bs = max(1, int(self.tsencoder_batch_size))

        if X.dim() == 3:
            B, T, L = X.shape
            x_flat = X.reshape(B * T, 1, L).to(device)
            reps = []
            with torch.no_grad():
                for i in range(0, x_flat.shape[0], bs):
                    reps.append(self.tsencoder_model(x_flat[i : i + bs]))
            reps = torch.cat(reps, dim=0)
            return reps.reshape(B, T, -1)

        if X.dim() == 4:
            B, T, C, L = X.shape
            x_flat = X.reshape(B * T * C, 1, L).to(device)
            reps = []
            with torch.no_grad():
                for i in range(0, x_flat.shape[0], bs):
                    reps.append(self.tsencoder_model(x_flat[i : i + bs]))
            reps = torch.cat(reps, dim=0)  # (B*T*C, D)
            reps = reps.reshape(B, T, C, -1).sum(dim=2)  # (B,T,D)
            return reps

        raise ValueError(f"Unexpected X dim for tsencoder encode: {X.dim()} with shape {tuple(X.shape)}")

    @torch.no_grad()
    def encode_and_adapt(self, X: torch.Tensor) -> torch.Tensor:
        reps = self._encode(X)
        reps = reps.to(X.device)
        return self.adapter(reps)

    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        d: torch.Tensor | None = None,
        feature_shuffles=None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config=None,
    ) -> torch.Tensor:
        # d/feature_shuffles/embed_with_test are ignored in this path.
        # NOTE: classifier_v2 runs in eval/no_grad outside; keep this forward lightweight.
        reps = self._encode(X)
        reps = reps.to(X.device)
        reps = self.adapter(reps)

        # Orion ICL accepts mgr_config (MgrConfig). We keep it None by default.
        mgr_config = None
        if inference_config is not None and hasattr(inference_config, "ICL_CONFIG"):
            # `inference_config` here is TIC.InferenceConfig. Its ICL_CONFIG type is
            # not guaranteed to match TIC.model.inference_config.MgrConfig, so we
            # intentionally do not pass it through.
            mgr_config = None

        return self.icl_predictor(
            reps,
            y_train=y_train,
            return_logits=bool(return_logits),
            softmax_temperature=float(softmax_temperature),
            mgr_config=mgr_config,
        )


@torch.no_grad()
def _predict_direct(
    model: _TSEncoderAdapterPlusOrionICL,
    *,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    query_batch_size: int,
    softmax_temperature: float,
) -> np.ndarray:
    """Directly predict labels for X_query using (support + query) ICL tables."""

    device = next(model.adapter.parameters()).device
    X_support_t = torch.from_numpy(X_support.astype(np.float32)).to(device)
    X_query_t = torch.from_numpy(X_query.astype(np.float32)).to(device)
    y_support_t = torch.from_numpy(y_support.astype(np.float32)).to(device)

    # support: (S,L) -> (1,S,L), query: (N,L) -> (B,1,L)
    X_support_t = X_support_t.unsqueeze(0)
    y_support_t = y_support_t.unsqueeze(0)

    support_rep = model.encode_and_adapt(X_support_t)  # (1,S,D)

    preds: list[np.ndarray] = []
    bs = max(1, int(query_batch_size))
    for i in range(0, X_query_t.shape[0], bs):
        q = X_query_t[i : i + bs].unsqueeze(1)  # (B,1,L)
        query_rep = model.encode_and_adapt(q)  # (B,1,D)

        B = query_rep.shape[0]
        reps_all = torch.cat([support_rep.repeat(B, 1, 1), query_rep], dim=1)  # (B,S+1,D)
        y_train = y_support_t.repeat(B, 1)  # (B,S)

        # Orion ICL forward returns logits/probs for test region only in eval mode.
        logits = model.icl_predictor(
            reps_all,
            y_train=y_train,
            return_logits=True,
            softmax_temperature=float(softmax_temperature),
        )
        # logits: (B, 1, num_classes)
        pred = torch.argmax(logits[:, 0, :], dim=-1).detach().cpu().numpy()
        preds.append(pred)

    return np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained adapter (tsencoder->adapter->OrionMSP icl_predictor) on UCR. "
            "This version loads the full _TSEncoderAdapterPlusOrionICL checkpoint + model_hparams JSON."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="classifier_v2",
        choices=["direct", "classifier_v2"],
        help=(
            "Evaluation mode: 'direct' uses cached support reps; "
            "'classifier_v2' uses TSEncoderICLClassifierV2 (class shift + RandomCropResize augmentation)."
        ),
    )

    parser.add_argument(
        "--full_ckpt",
        type=str,
        default="/data0/fangjuntao2025/TIS-FS_Open/checkpoints/TSEncoder_orion_icl_full.pt",
        help="Path to full _TSEncoderAdapterPlusOrionICL checkpoint (.pt).",
    )
    parser.add_argument(
        "--model_hparams_json",
        type=str,
        default="/data0/fangjuntao2025/TIS-FS_Open/checkpoints/TSEncoder_orion_icl_model_hparams.json",
        help="Path to model hyperparameters JSON used to rebuild the full model.",
    )
    parser.add_argument(
        "--save_model_hparams_json",
        type=str,
        default=None,
        help="Optional path to save inferred _TSEncoderAdapterPlusOrionICL hyperparameters JSON.",
    )

    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument(
        "--uea_path",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
        help="Only for DataReader initialization; UCR evaluation does not require UEA datasets.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--suite",
        type=str,
        default="ucr",
        choices=["ucr", "uea"],
        help="Dataset suite to evaluate: UCR (univariate) or UEA (multivariate).",
    )

    parser.add_argument("--dataset", type=str, default=None, help="Evaluate a single UCR dataset name")

    parser.add_argument("--n_estimators", type=int, default=32)

    v2_class_shift_group = parser.add_mutually_exclusive_group()
    v2_class_shift_group.add_argument(
        "--v2_class_shift",
        dest="v2_class_shift",
        action="store_true",
        default=True,
        help="(classifier_v2) Enable class shift ensembling. Default: True.",
    )
    v2_class_shift_group.add_argument(
        "--no_v2_class_shift",
        dest="v2_class_shift",
        action="store_false",
        help="(classifier_v2) Disable class shift ensembling.",
    )
    parser.add_argument(
        "--v2_n_augmentations",
        type=int,
        default=1,
        help="(classifier_v2) Number of RandomCropResize views per estimator.",
    )
    parser.add_argument(
        "--v2_crop_rate_lo",
        type=float,
        default=0.0,
        help="(classifier_v2) RandomCropResize crop_rate lower bound.",
    )
    parser.add_argument(
        "--v2_crop_rate_hi",
        type=float,
        default=0.00,
        help="(classifier_v2) RandomCropResize crop_rate upper bound.",
    )

    parser.add_argument(
        "--support_size",
        type=int,
        default=128,
        help="Number of training samples used as ICL support (auto-bumped to >= #classes).",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=64,
        help="Query batch size (number of test samples per forward).",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=0.9,
        help="Softmax temperature passed to icl_predictor inference.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for support sampling.")

    parser.add_argument(
        "--uea_mode",
        type=str,
        default="center",
        choices=["center", "random"],
        help="(UEA) After channel-concat, how to sample length=tsencoder_seq_len window.",
    )

    parser.add_argument(
        "--uea_use_var_selector",
        action="store_true",
        help="(UEA) Use VarianceBasedSelector to reduce channels before inference.",
    )
    parser.add_argument(
        "--uea_var_num_channels",
        type=int,
        default=None,
        help="(UEA) Target number of channels after VarianceBasedSelector.",
    )
    parser.add_argument(
        "--uea_fusion",
        type=str,
        default="concat",
        choices=["concat", "sum_embed"],
        help=(
            "(UEA) How to handle multichannel series: "
            "'concat' concatenates channels in time then samples length=tsencoder_seq_len; "
            "'sum_embed' encodes each channel with TSEncoder then sums channel embeddings."
        ),
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model hyperparameters JSON
    model_hparams = _load_model_hparams_json(str(args.model_hparams_json))

    # Build TSEncoder (no ckpt load; weights come from full checkpoint)
    tsencoder_cfg = model_hparams.get("tsencoder", {})
    tsencoder_hidden_dim = int(tsencoder_cfg.get("hidden_dim", 512))
    tsencoder_seq_len = int(tsencoder_cfg.get("seq_len", 512))
    tsencoder_dim = int(tsencoder_cfg.get("tsencoder_dim", tsencoder_hidden_dim))

    tsencoder_model = build_TSEncoder(
        tsencoder_checkpoint=None,
        device=device,
        hidden_dim=int(tsencoder_hidden_dim),
        seq_len=int(tsencoder_seq_len),
    )
    for p in tsencoder_model.parameters():
        p.requires_grad_(False)
    tsencoder_model.eval()

    # Build Adapter from JSON
    adapter_cfg = model_hparams.get("adapter", {})
    icl_dim = int(adapter_cfg.get("icl_dim", 512))
    adapter_hidden_dim_val = adapter_cfg.get("hidden_dim")
    adapter_dropout_val = float(adapter_cfg.get("dropout", 0.0))
    adapter_use_ln = bool(adapter_cfg.get("use_layernorm", True))

    adapter = TokenMLPAdapter(
        tsencoder_dim=int(tsencoder_dim),
        icl_dim=int(icl_dim),
        hidden_dim=(None if adapter_hidden_dim_val is None else int(adapter_hidden_dim_val)),
        dropout=float(adapter_dropout_val),
        use_layernorm=bool(adapter_use_ln),
    )
    adapter.to(device)
    adapter.eval()

    # Build ICL predictor from JSON
    icl_predictor = _build_icl_predictor_from_hparams(model_hparams)
    for p in icl_predictor.parameters():
        p.requires_grad_(False)
    icl_predictor.to(device)
    icl_predictor.eval()

    custom_model = _TSEncoderAdapterPlusOrionICL(
        tsencoder_model=tsencoder_model,
        adapter=adapter,
        icl_predictor=icl_predictor,
        tsencoder_seq_len=int(tsencoder_seq_len),
        tsencoder_batch_size=int(tsencoder_cfg.get("batch_size", 64)),
    )
    for p in custom_model.parameters():
        p.requires_grad_(False)
    custom_model.eval()

    # Load full checkpoint state_dict
    ckpt_obj = torch.load(str(args.full_ckpt), map_location="cpu")
    state_dict = _extract_full_state_dict(ckpt_obj)
    custom_model.load_state_dict(state_dict, strict=True)

    # Print PerceiverMemory config
    orion_cfg = model_hparams.get("orion", {}).get("icl_predictor", {}).get("config", {})
    perc_num_latents = orion_cfg.get("perc_num_latents", None)
    print(f"[Orion Config] perc_num_latents = {perc_num_latents}")
    if perc_num_latents is not None and int(perc_num_latents) > 0:
        print("[Orion Config] PerceiverMemory 已启用 (推理时 ICL 会用 Memory)")
    else:
        print("[Orion Config] PerceiverMemory 未启用 (推理时 ICL 不用 Memory)")

    embed_dim = int(orion_cfg.get("embed_dim", 128))
    row_num_cls = int(icl_dim // embed_dim) if embed_dim > 0 and icl_dim % embed_dim == 0 else 0

    # 输出完整模型结构与各模块超参数（从 JSON + full ckpt 读取后）
    print("[Eval] _TSEncoderAdapterPlusOrionICL structure:")
    print(custom_model)
    model_hparams_out = _collect_model_hparams(
        custom_model=custom_model,
        tsencoder_ckpt=str(tsencoder_cfg.get("ckpt", "")),
        orion_ckpt=str(model_hparams.get("orion", {}).get("ckpt", "")),
        tsencoder_dim=int(tsencoder_dim),
        icl_dim=int(icl_dim),
        adapter_hidden_dim_val=(None if adapter_hidden_dim_val is None else int(adapter_hidden_dim_val)),
        adapter_dropout_val=float(adapter_dropout_val),
        adapter_no_ln_val=bool(not adapter_use_ln),
        orion_cfg=orion_cfg,
        perc_num_latents=(None if perc_num_latents is None else int(perc_num_latents)),
        embed_dim=int(embed_dim),
        row_num_cls=int(row_num_cls),
    )
    _print_effective_runtime_config(title="[Eval][ModelHyperparams]", payload=model_hparams_out)

    if args.save_model_hparams_json:
        out_path = Path(args.save_model_hparams_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_hparams_out, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        print(f"[Eval] Saved model hyperparameters to {out_path}")

    reader = DataReader(
        UEA_data_path=str(args.uea_path),
        UCR_data_path=str(args.ucr_path),
        transform_ts_size=int(tsencoder_seq_len),
    )

    if args.dataset is not None:
        dataset_names = [args.dataset]
    else:
        if args.suite == "uea":
            dataset_names = list(reader.dataset_list_uea)
        else:
            dataset_names = list(reader.dataset_list_ucr)

    print(f"[Eval] full_ckpt: {args.full_ckpt}")
    print(f"[Eval] model_hparams_json: {args.model_hparams_json}")
    print(f"[Eval] suite: {args.suite}")
    print(f"[Eval] dataset_count: {len(dataset_names)}")
    print(f"[Eval] mode: {args.mode}")

    # Print the effective runtime configuration
    _print_effective_runtime_config(
        title="[Eval][EffectiveRuntimeConfig]",
        payload={
            "cli_args": vars(args),
            "resolved": {
                "full_ckpt": str(args.full_ckpt),
                "model_hparams_json": str(args.model_hparams_json),
                "tsencoder_dim": int(tsencoder_dim),
                "icl_dim": int(icl_dim),
                "adapter_hidden_dim": (None if adapter_hidden_dim_val is None else int(adapter_hidden_dim_val)),
                "adapter_dropout": float(adapter_dropout_val),
                "adapter_use_layernorm": bool(adapter_use_ln),
                "dataset_names_count": int(len(dataset_names)),
                "dataset_single": (None if args.dataset is None else str(args.dataset)),
            },
        },
    )

    accs: list[float] = []
    for name in dataset_names:
        try:
            X_tr, y_tr = reader.read_dataset(name, which_set="train")
            X_te, y_te = reader.read_dataset(name, which_set="test")

            if args.suite == "uea":
                X_tr, X_te = _maybe_select_channels_uea(
                    X_tr,
                    X_te,
                    enabled=bool(args.uea_use_var_selector),
                    new_num_channels=(None if args.uea_var_num_channels is None else int(args.uea_var_num_channels)),
                    dataset_name=name,
                )

                if args.uea_fusion == "sum_embed":
                    # Keep multichannel and crop/pad per channel to tsencoder_seq_len.
                    X_tr_2d = _uea_crop_pad_per_channel(
                        X_tr,
                        target_len=int(tsencoder_seq_len),
                        seed=int(args.seed),
                        mode=str(args.uea_mode),
                    )
                    X_te_2d = _uea_crop_pad_per_channel(
                        X_te,
                        target_len=int(tsencoder_seq_len),
                        seed=int(args.seed) + 1,
                        mode=str(args.uea_mode),
                    )
                else:
                    # Backward-compatible: concatenate channels then sample/crop.
                    X_tr_2d = _uea_concat_and_sample(
                        X_tr,
                        target_len=int(tsencoder_seq_len),
                        seed=int(args.seed),
                        mode=str(args.uea_mode),
                    )
                    X_te_2d = _uea_concat_and_sample(
                        X_te,
                        target_len=int(tsencoder_seq_len),
                        seed=int(args.seed) + 1,
                        mode=str(args.uea_mode),
                    )
            else:
                X_tr_2d = _ensure_2d_timeseries(X_tr)
                X_te_2d = _ensure_2d_timeseries(X_te)

                # UCR should already be (N,L); if UEA slips in here, fail loudly.
                if X_tr_2d.ndim != 2 or X_te_2d.ndim != 2:
                    raise ValueError(
                        f"Expected univariate (N,L) arrays for UCR; got train {X_tr_2d.shape}, test {X_te_2d.shape}. "
                        f"Did you mean --suite uea ?"
                    )

            y_tr_m, y_te_m, _classes = _remap_labels(y_tr, y_te)

            if args.mode == "classifier_v2":
                if args.suite == "uea" and args.uea_fusion == "sum_embed":
                    raise ValueError(
                        "classifier_v2 expects 2D inputs (n_samples, length). "
                        "For UEA multichannel sum_embed, use --mode direct (or set --uea_fusion concat)."
                    )
                crop_lo = float(args.v2_crop_rate_lo)
                crop_hi = float(args.v2_crop_rate_hi)
                if not (0.0 <= crop_lo <= crop_hi < 1.0):
                    raise ValueError(
                        f"Invalid v2 crop range: lo={crop_lo}, hi={crop_hi}. Must satisfy 0 <= lo <= hi < 1."
                    )

                clf = TSEncoderICLClassifierV2(
                    n_estimators=int(args.n_estimators),
                    class_shift=bool(args.v2_class_shift),
                    crop_rate_range=(crop_lo, crop_hi),
                    n_augmentations=int(args.v2_n_augmentations),
                    softmax_temperature=float(args.softmax_temperature),
                    device=device,
                    random_state=int(args.seed),
                    verbose=False,
                    model_path=None,
                    allow_auto_download=False,
                    checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                )
                clf.model_ = custom_model
                clf.fit(X_tr_2d, y_tr_m)
                y_pred = clf.predict(X_te_2d)
                acc = float(np.mean(y_pred == y_te_m))
            else:
                sup_idx = _select_support_indices(y_tr_m, support_size=int(args.support_size), seed=int(args.seed))
                X_sup = X_tr_2d[sup_idx]
                y_sup = y_tr_m[sup_idx]

                y_pred = _predict_direct(
                    custom_model,
                    X_support=X_sup,
                    y_support=y_sup,
                    X_query=X_te_2d,
                    query_batch_size=int(args.query_batch_size),
                    softmax_temperature=float(args.softmax_temperature),
                )
                acc = float(np.mean(y_pred == y_te_m))

            print(f"{name}: {acc:.4f}")
            accs.append(acc)
        except Exception as e:
            print(f"{name}: failed: {e}")

    if accs:
        print(f"\nEvaluated {len(accs)} {args.suite.upper()} datasets | mean accuracy: {float(np.mean(accs)):.4f}")
    else:
        print("No datasets evaluated successfully.")


if __name__ == "__main__":
    main()
