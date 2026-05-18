import numpy as np
from scipy.stats import theilslopes


def estimate_pitch_from_widths(widths, f_x, f_y, image_height, w_real,
                               return_debug: bool = False):
    """Estimate camera-to-road-plane pitch from per-band lane widths.

    Parameters
    ----------
    widths : np.ndarray of shape (N, 2)
        (y_pixel, pixel_width) pairs sampled along the visible road.
    f_x, f_y : float
        Camera focal lengths (pixels) for the resized image.
    image_height : int
        Image height in pixels (for principal-point computation).
    w_real : float
        Real-world lane width (m).
    return_debug : bool
        If True, return ``(pitch_deg, depths, Y_3d)`` instead of just
        ``pitch_deg``. The extra arrays correspond to the IQR-filtered samples
        actually used by Theil-Sen, so they capture the depth range the
        algorithm "saw" on this frame. Default False (backward-compatible).
    """
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        raise ValueError(
            "estimate_pitch_from_widths: widths must be shape (N, 2) with N > 0, "
            "got shape {}. Left/right lane fits likely have no overlapping y-range "
            "or produced no valid sample points.".format(widths.shape)
        )

    # IQR outlier filter on pixel width
    w = widths[:, 1]
    q1, q3 = np.percentile(w, [25, 75])
    iqr = q3 - q1
    valid = (w >= q1 - 1.5 * iqr) & (w <= q3 + 1.5 * iqr)
    widths = widths[valid]

    if len(widths) == 0:
        raise ValueError(
            "estimate_pitch_from_widths: all {} sample(s) were removed by the "
            "IQR filter — lane widths too inconsistent to estimate pitch.".format(len(w))
        )

    depths = f_x * w_real / widths[:, 1]
    center_y = image_height / 2
    Y_3d = -depths * (widths[:, 0] - center_y) / f_y

    result = theilslopes(Y_3d, depths)
    pitch_rad = np.arctan(result.slope)
    pitch_deg = float(np.degrees(pitch_rad))

    if return_debug:
        return pitch_deg, depths, Y_3d
    return pitch_deg
