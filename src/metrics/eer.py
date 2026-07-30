"""EER computation for the ASVspoof2019 LA countermeasure.

The EER itself is computed by `compute_eer` from `src/metrics/calculate_eer.py`,
which is the official routine supplied with the assignment. This module only
wraps it: it converts the fraction returned by `compute_eer` into percent and
exposes a BaseMetric subclass.

Note that EER is a set-level metric: it must be computed over all trials of a
partition at once, never averaged over mini-batches. The pooled value is
computed by the trainer; `EERMetric` below is kept only for completeness and is
not registered in the metrics config.
"""

import numpy as np
import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.calculate_eer import compute_det_curve, compute_eer

__all__ = ["compute_det_curve", "compute_eer", "compute_eer_numpy", "EERMetric"]


def compute_eer_numpy(bonafide_scores, spoof_scores):
    """Official compute_eer, with the EER expressed in percent (0-100 scale),
    as required by the grading script."""
    eer, threshold = compute_eer(
        np.asarray(bonafide_scores, dtype=np.float64),
        np.asarray(spoof_scores, dtype=np.float64),
    )
    return float(eer) * 100.0, float(threshold)


def scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Countermeasure score for each trial.

    The evaluation plan requires a high score to indicate bona fide and a low
    score to indicate a spoofing attack, so we use the difference of the two
    logits (class 1 = bona fide, class 0 = spoof).
    """
    return logits[:, 1] - logits[:, 0]


class EERMetric(BaseMetric):
    def __init__(self, name="eer", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        logits = logits.detach().cpu()
        labels = labels.detach().cpu().long().numpy()

        scores = scores_from_logits(logits).numpy()

        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]

        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            return 0.0

        eer, _ = compute_eer_numpy(bonafide_scores, spoof_scores)
        return float(eer)

