import numpy as np
import cv2
# from sklearn.linear_model import RANSACRegressor

"""
Use points view to do piecewise linear fitting
"""
def collect_points_from_segments(segments, extra_points_per_segment):
    points = []

    for seg in segments:
        x1, y1, x2, y2 = seg
        """
        Add the extra points along the line segment to make the fitting more robust
        """
        for t in np.linspace(0, 1, extra_points_per_segment):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            points.append((x, y))
    return np.array(points)  # Output shape: (N, 2), list of (x, y) points

def piecewise_linear_fit(points, num_bands):

    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    band_edges = np.linspace(y_min, y_max, num_bands + 1)

    fits = []
    for i in range(num_bands):

        y_lower_bound, y_upper_bound = band_edges[i], band_edges[i + 1]
        mask = (points[:, 1] >= y_lower_bound) & (points[:, 1] < y_upper_bound)
        band_points = points[mask]

        if len(band_points) < 2:
            continue

        """
        RANSAC was tried here but made results worse: far bands have few points with
        natural perspective spread, so a fixed residual_threshold excludes too many
        valid points and causes the fit to go off.
        """
        # ransac = RANSACRegressor(min_samples=2, residual_threshold=ransac_residual_threshold, random_state=42)
        # ransac.fit(band_points[:, 1].reshape(-1, 1), band_points[:, 0])
        # a, b = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
        coeffs = np.polyfit(band_points[:, 1], band_points[:, 0], deg=1)
        a, b = coeffs

        fits.append({
            "y_start": y_lower_bound,
            "y_end": y_upper_bound,
            "slope": a,
            "intercept": b,
            "num_points": len(band_points)
        })

    return fits
"""
get x-coordinate at a specific y band based on the piecewise linear fits
"""
def get_x_at_y(fits, y):
    for f in fits:
        if f["y_start"] <= y <= f["y_end"]:
            return f["slope"] * y + f["intercept"]
    return None

def compute_lane_widths(left_fits, right_fits, num_samples):

    """
    Find the overlapping y-range of the left and right fits
    """
    left_y_min = min(f["y_start"] for f in left_fits)
    left_y_max = max(f["y_end"] for f in left_fits)
    right_y_min = min(f["y_start"] for f in right_fits)
    right_y_max = max(f["y_end"] for f in right_fits)

    y_min = max(left_y_min, right_y_min)
    y_max = min(left_y_max, right_y_max)

    """
    Sample num_samples y-values within the overlapping range
    """
    sample_ys = np.linspace(y_min, y_max, num_samples)
    widths = []

    for y in sample_ys:
        x_left = get_x_at_y(left_fits, y)
        x_right = get_x_at_y(right_fits, y)
        if x_left is not None and x_right is not None:
            widths.append((y, x_right - x_left))

    # Always return shape (N, 2) so callers can safely do widths[:, 0] / widths[:, 1].
    # np.array([]) produces shape (0,) which causes IndexError downstream.
    if not widths:
        return np.empty((0, 2))
    return np.array(widths)  # shape (N, 2)
