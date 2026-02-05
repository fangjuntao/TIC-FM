from __future__ import annotations

import warnings
import random
from pathlib import Path
from packaging import version
from typing import Optional, Dict

import numpy as np
import torch

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder

from .preprocessing import TransformToNumerical
from TIC.model.inference_config import InferenceConfig
from TIC.model.TSEncoder_dev.trainer.trainer_utils.pretraining import RandomCropResize

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
OLD_SKLEARN = version.parse(sklearn.__version__) < version.parse("1.6")


class TSEncoderICLClassifierV2(ClassifierMixin, BaseEstimator):
    """A simplified TSEncoderICL sklearn classifier (v2).

    Notes
    -----
    This is primarily intended for time-series-as-tabular inputs (e.g. UCR), where
    each sample is a length-L series represented as a 2D array (n_samples, L).

    If you want to evaluate a custom model (e.g. tsencoder->adapter->icl_predictor),
    you may set `clf.model_ = your_model` before calling `fit()`.
    """

    def __init__(
        self,
        n_estimators: int = 32,
        class_shift: bool = True,
        crop_rate_range: tuple[float, float] = (0.0, 0.2),
        n_augmentations: int = 2,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        use_hierarchical: bool = True,
        use_amp: bool = True,
        batch_size: Optional[int] = 8,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
        device: Optional[str | torch.device] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
    ):
        self.n_estimators = n_estimators
        self.class_shift = class_shift
        self.crop_rate_range = crop_rate_range
        self.n_augmentations = n_augmentations
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.use_hierarchical = use_hierarchical
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.inference_config = inference_config

    def _more_tags(self):
        return dict(non_deterministic=True)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    def _load_model(self):
        raise RuntimeError(
            "Default TabICL checkpoint loading has been removed. "
            "Please set `clf.model_` to a compatible model before calling fit()."
        )

    def fit(self, X, y):
        if OLD_SKLEARN:
            X, y = self._validate_data(X, y, dtype=None, cast_to_ndarray=False)
        else:
            X, y = self._validate_data(X, y, dtype=None, skip_check_array=True)

        check_classification_targets(y)

        if self.device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device

        need_load = not hasattr(self, "model_")
        if not need_load and self.model_path is not None:
            try:
                current_path = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
                if getattr(self, "model_path_", None) != current_path:
                    need_load = True
            except Exception:
                need_load = True
        if need_load:
            self._load_model()
        self.model_.to(self.device_)

        init_config = {
            "COL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
            "ROW_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
            "ICL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
        }
        if self.inference_config is None:
            self.inference_config_ = InferenceConfig()
            self.inference_config_.update_from_dict(init_config)
        elif isinstance(self.inference_config, dict):
            self.inference_config_ = InferenceConfig()
            for key, value in self.inference_config.items():
                if key in init_config:
                    init_config[key].update(value)
            self.inference_config_.update_from_dict(init_config)
        else:
            self.inference_config_ = self.inference_config

        self.y_encoder_ = LabelEncoder()
        y = self.y_encoder_.fit_transform(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.y_encoder_.classes_)

        if self.n_classes_ > self.model_.max_classes and not self.use_hierarchical:
            raise ValueError(
                f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                f"natively supported by the model. Consider enabling hierarchical classification."
            )

        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X_num = self.X_encoder_.fit_transform(X)
        X_num = np.asarray(X_num, dtype=np.float32)

        self.X_train_ = X_num
        self.y_train_ = np.asarray(y, dtype=np.int64)
        self.n_features_in_ = self.X_train_.shape[1]

        # RNG for class shift ordering / crop rates
        self.rng_ = random.Random(self.random_state)
        return self

    def _make_members(self) -> list[tuple[int, float]]:
        """Return list of (class_shift_offset, crop_rate) members."""
        n_estimators = max(1, int(self.n_estimators))
        n_aug = max(1, int(self.n_augmentations))

        if self.class_shift and n_estimators > 1:
            base_offsets = list(range(self.n_classes_))
            self.rng_.shuffle(base_offsets)
            offsets = [base_offsets[i % len(base_offsets)] for i in range(n_estimators)]
        else:
            offsets = [0 for _ in range(n_estimators)]

        lo, hi = self.crop_rate_range
        lo = float(lo)
        hi = float(hi)
        if not (0.0 <= lo <= hi < 1.0):
            raise ValueError(f"crop_rate_range must satisfy 0 <= lo <= hi < 1, got {self.crop_rate_range}")

        members: list[tuple[int, float]] = []
        for off in offsets:
            for _ in range(n_aug):
                crop_rate = lo if lo == hi else (lo + (hi - lo) * self.rng_.random())
                members.append((int(off), float(crop_rate)))
        return members

    def _batch_forward_members(
        self,
        X_all: np.ndarray,
        y_train: np.ndarray,
        members: list[tuple[int, float]],
    ) -> tuple[np.ndarray, list[int]]:
        """Forward all ensemble members with augmentation and class shift.

        Returns
        -------
        outputs: np.ndarray
            Shape (n_members, test_size, n_classes)
        offsets: list[int]
            Class shift offset per member (same length as n_members)
        """

        device = self.device_
        train_size = int(y_train.shape[0])
        test_size = int(X_all.shape[0] - train_size)
        if test_size <= 0:
            raise ValueError("X_test must have at least 1 sample")

        # Prepare base tensor once
        x_all_t = torch.from_numpy(X_all.astype(np.float32)).to(device).unsqueeze(1)  # (N, 1, L)

        member_offsets: list[int] = [m[0] for m in members]
        outputs: list[np.ndarray] = []

        bs = self.batch_size or len(members)
        bs = max(1, int(bs))
        for start in range(0, len(members), bs):
            chunk = members[start : start + bs]
            Xs = []
            ys = []
            chunk_offsets: list[int] = []
            for off, crop_rate in chunk:
                x_aug = RandomCropResize(x_all_t, crop_rate=float(crop_rate))  # (N, 1, L)
                x_aug = x_aug.squeeze(1)  # (N, L)
                Xs.append(x_aug)
                y_shift = ((y_train + off) % self.n_classes_).astype(np.float32)
                ys.append(torch.from_numpy(y_shift).to(device))
                chunk_offsets.append(int(off))

            X_batch = torch.stack(Xs, dim=0)  # (B, N, L)
            y_batch = torch.stack(ys, dim=0)  # (B, train_size)

            with torch.no_grad():
                out = self.model_(
                    X_batch,
                    y_batch,
                    feature_shuffles=None,
                    return_logits=True if self.average_logits else False,
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            out_np = out.float().cpu().numpy()
            if out_np.shape[1] != test_size:
                raise RuntimeError(f"Unexpected model output shape: {out_np.shape}, expected test_size={test_size}")
            outputs.append(out_np)

        outputs_np = np.concatenate(outputs, axis=0)
        return outputs_np, member_offsets

    def predict_proba(self, X):
        check_is_fitted(self)

        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")

        if self.n_jobs is not None:
            assert self.n_jobs != 0
            old_n_threads = torch.get_num_threads()

            import multiprocessing as mp

            n_logical_cores = mp.cpu_count()

            if self.n_jobs > 0:
                if self.n_jobs > n_logical_cores:
                    warnings.warn(
                        f"TSEncoderICLClassifierV2 got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                        f" Only {n_logical_cores} threads will be used."
                    )
                n_threads = min(n_logical_cores, self.n_jobs)
            else:
                n_threads = max(1, mp.cpu_count() + 1 + self.n_jobs)

            torch.set_num_threads(n_threads)

        if OLD_SKLEARN:
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)

        X_num = self.X_encoder_.transform(X)
        X_num = np.asarray(X_num, dtype=np.float32)

        X_all = np.concatenate([self.X_train_, X_num], axis=0)
        members = self._make_members()

        outputs, offsets = self._batch_forward_members(X_all, self.y_train_, members)

        # Aggregate predictions, reversing class shifts
        avg = None
        for out, off in zip(outputs, offsets):
            if off != 0:
                out = np.concatenate([out[..., off:], out[..., :off]], axis=-1)
            if avg is None:
                avg = out
            else:
                avg += out

        avg /= len(offsets)

        if self.average_logits:
            avg = self.softmax(avg, axis=-1, temperature=self.softmax_temperature)

        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)

        return avg / avg.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)
        return self.y_encoder_.inverse_transform(y)

    @staticmethod
    def softmax(x, axis: int = -1, temperature: float = 0.9):
        x = x / temperature
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)




