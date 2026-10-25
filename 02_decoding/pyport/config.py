"""Decoder — Python readout (02_decoding/pyport)

Decoder configuration; the defaults are the readout settings and each published run overrides some of them with a file in configs/

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import json
from dataclasses import dataclass, field, asdict

import numpy as np

from .mcompat import parse_grid, mround

LAMBDA_GRID = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]


@dataclass
class Config:
    pca_k: int = 20
    lag_ms: object = '-2000:50:0'
    post_fir_lag_ms: object = '-2000:50:0'
    post_fir_causal_only: bool = True
    lambda_grid: list = field(default_factory=lambda: list(LAMBDA_GRID))
    post_lambda_grid: list = field(default_factory=lambda: list(LAMBDA_GRID))
    outer_k: int = 10
    outer_folds: object = None
    inner_k: int = 10
    post_inner_k: int = 10
    eeg_baseline_mode: str = 'mean'
    mode: str = 'fast'
    split: str = 'random'
    chrono_calib_blocks: int = 3
    input_control: str = 'full'
    input_control_seed: int = 20260905
    cross_scope: str = 'folds'
    stage1_lpf_hz: float = 0.0

    @classmethod
    def from_json(cls, path_or_dict):
        d = path_or_dict
        if isinstance(d, str):
            with open(d) as fh:
                d = json.load(fh)
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('unknown config keys: %s' % sorted(unknown))
        return cls(**d)

    def to_dict(self):
        return asdict(self)

    def stage1_lags(self, fs):
        lags = np.unique(mround(parse_grid(self.lag_ms) * fs / 1000))
        if lags.size == 0 or lags.max() > 0:
            raise ValueError('stage-1 lags must be non-empty and <= 0 (export window invariant)')
        return lags.astype(np.int64)

    def stage2_lags(self, fs):
        lags = parse_grid(self.post_fir_lag_ms)
        lags = lags[np.isfinite(lags)]
        if lags.size == 0:
            lags = colon_default()
        lags = np.unique(mround(lags * float(fs) / 1000))
        if self.post_fir_causal_only:
            lags = lags[lags <= 0]
        if lags.size == 0:
            raise ValueError('post FIR has no causal lags')
        return lags.astype(np.int64)


def colon_default():
    from .mcompat import colon
    return colon(-400, 40, 0)
