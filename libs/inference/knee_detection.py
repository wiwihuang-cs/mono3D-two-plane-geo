"""Trend-based detector for dynamic positive/negative angle lane segmentation.

Aggregation policy (this revision):
  * detect_side_knee returns direction=0 when t-stat is low OR effect size
    abs(near_abs - far_abs) < min_slope_diff.
  * detect_knee commits to a version ONLY when both sides commit AND agree.
    Anything else (one weak, both weak, or both committed but disagreeing)
    falls back to ``either``, with a diagnostic ``reason`` tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SideKnee:
    knee_y: Optional[float]
    direction: int
    confidence: float
    near_abs_slope: float
    far_abs_slope: float
    n_segments: int

    @classmethod
    def empty(cls) -> "SideKnee":
        return cls(None, 0, 0.0, 0.0, 0.0, 0)


@dataclass
class KneeResult:
    version: str
    knee_y: Optional[float]
    direction: int
    confidence: float
    left: SideKnee
    right: SideKnee
    reason: str
    n_left_raw: int = 0
    n_right_raw: int = 0


def _segments_to_features(segments):
    if len(segments) == 0:
        empty = np.empty(0)
        return empty, empty, empty
    arr = np.asarray(segments, dtype=np.float64)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    valid = x1 != x2
    arr = arr[valid]
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    slope = (y2 - y1) / (x2 - x1)
    bottom_y = np.maximum(y1, y2)
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    order = np.argsort(-bottom_y)
    return bottom_y[order], slope[order], length[order]


def weighted_linregress(x, y, w):
    sw = w.sum()
    if sw <= 0:
        return 0.0, 0.0, 0.0
    xm = (w * x).sum() / sw
    ym = (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    sxy = (w * (x - xm) * (y - ym)).sum()
    if sxx <= 1e-9:
        return 0.0, float(ym), 0.0
    a = sxy / sxx
    b = ym - a * xm
    resid = y - (a * x + b)
    n_eff = (sw ** 2) / (w ** 2).sum()
    if n_eff <= 2:
        return float(a), float(b), 0.0
    sigma2 = (w * resid ** 2).sum() / (sw * (n_eff - 2) / n_eff)
    var_a = sigma2 / sxx
    tstat = abs(a) / (np.sqrt(var_a) + 1e-12)
    return float(a), float(b), float(tstat)


def detect_side_knee(
    segments,
    *,
    min_segments: int = 4,
    min_tstat: float = 2.0,
    min_slope_diff: float = 0.05,
    smooth_window: int = 5,
):
    """Trend-based per-side detector. Returns direction=0 when signal is weak."""
    del smooth_window

    bottom_y, slope, length = _segments_to_features(segments)
    n = bottom_y.size
    if n < min_segments:
        return SideKnee.empty()

    abs_slope = np.abs(slope)

    median_y = float(np.median(bottom_y))
    near_mask = bottom_y >= median_y
    far_mask = ~near_mask
    near_abs = float(np.average(abs_slope[near_mask], weights=length[near_mask])) \
        if near_mask.any() else 0.0
    far_abs = float(np.average(abs_slope[far_mask], weights=length[far_mask])) \
        if far_mask.any() else 0.0

    a, _b, tstat = weighted_linregress(bottom_y, abs_slope, length)

    if tstat < min_tstat or abs(near_abs - far_abs) < min_slope_diff:
        return SideKnee(
            knee_y=None, direction=0, confidence=0.0,
            near_abs_slope=near_abs, far_abs_slope=far_abs, n_segments=n,
        )

    direction = +1 if a < 0 else -1
    confidence = float(1.0 / (1.0 + np.exp(-(tstat - min_tstat))))

    return SideKnee(
        knee_y=None, direction=direction, confidence=confidence,
        near_abs_slope=near_abs, far_abs_slope=far_abs, n_segments=n,
    )


def detect_knee(left_segments, right_segments, *, knee_y_tolerance: float = 30.0, **side_kwargs):
    """Strict aggregation: commit ONLY when both sides commit AND agree.

    Otherwise return ``either`` with a diagnostic ``reason``:
      * ``both_sides_weak``  — neither side committed
      * ``left_weak``        — only right committed
      * ``right_weak``       — only left committed
      * ``sides_disagree``   — both committed but to opposite directions
    """
    del knee_y_tolerance
    left = detect_side_knee(left_segments, **side_kwargs)
    right = detect_side_knee(right_segments, **side_kwargs)
    n_left_raw = len(left_segments)
    n_right_raw = len(right_segments)

    if (left.direction != 0 and right.direction != 0
            and left.direction == right.direction):
        version = "positive_angle" if left.direction == +1 else "negative_angle"
        conf = 0.5 * (left.confidence + right.confidence)
        return KneeResult(version=version, knee_y=None, direction=left.direction,
                          confidence=conf, left=left, right=right,
                          reason="both_sides_agree",
                          n_left_raw=n_left_raw, n_right_raw=n_right_raw)

    if left.direction == 0 and right.direction == 0:
        reason = "both_sides_weak"
    elif left.direction == 0:
        reason = "left_weak"
    elif right.direction == 0:
        reason = "right_weak"
    else:
        reason = "sides_disagree"

    return KneeResult(version="either", knee_y=None, direction=0, confidence=0.0,
                      left=left, right=right, reason=reason,
                      n_left_raw=n_left_raw, n_right_raw=n_right_raw)


class HysteresisVoter:
    def __init__(self, window: int = 5, default: str = "positive_angle"):
        self.window = window
        self.default = default
        self._history: List[str] = []
        self.last_output: str = default

    def update(self, result) -> str:
        # (A) Fallback: on 'either', reuse last output instead of a fixed default
        if result.version != "either" and result.confidence >= 0.4:
            self._history.append(result.version)
        else:
            self._history.append(self.last_output)

        if len(self._history) > self.window:
            self._history = self._history[-self.window:]

        cnt_pos = self._history.count("positive_angle")
        cnt_neg = self._history.count("negative_angle")

        # (B) Tie-break: keep last output when votes are equal
        if cnt_pos == cnt_neg:
            current_output = self.last_output
        else:
            current_output = "positive_angle" if cnt_pos > cnt_neg else "negative_angle"

        # (C) Persist result for next frame
        self.last_output = current_output
        return current_output


# ---------------------------------------------------------------------------
# Shared loose split — single source of truth for knee detection pre-filter
# ---------------------------------------------------------------------------

def loose_split(
    segments: np.ndarray,
    image_width: int,
    image_height: int,
    min_slope: float,
) -> Tuple[list, list]:
    """Assign ELSED segments to left/right based on slope sign and half-image position.

    No ROI is applied — the filter is intentionally permissive so that
    ``detect_knee`` receives as many slope samples as possible.

    A segment qualifies as:
      left  : slope < 0  AND  mid_x <= image_width / 2
      right : slope > 0  AND  mid_x >= image_width / 2

    Parameters
    ----------
    segments : np.ndarray, shape (N, 4)
        Raw ELSED output: each row is (x1, y1, x2, y2).
    image_width : int
    image_height : int
        Passed for API compatibility; not used in the no-ROI version.
    min_slope : float
        Segments whose |slope| < min_slope are discarded (near-horizontal).

    Returns
    -------
    left_raw, right_raw : list of (x1, y1, x2, y2) float tuples
    """
    left: list = []
    right: list = []
    cx = image_width * 0.5
    for x1, y1, x2, y2 in segments.astype(np.int32):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < min_slope:
            continue
        mid_x = (x1 + x2) * 0.5
        if slope < 0 and mid_x <= cx:
            left.append((float(x1), float(y1), float(x2), float(y2)))
        elif slope > 0 and mid_x >= cx:
            right.append((float(x1), float(y1), float(x2), float(y2)))
    return left, right
