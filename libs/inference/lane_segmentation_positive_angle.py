import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

"""
Split the line segments into left and right lane candidates based on their slope and position in the image.
And finally sort the line segments based on their x-coordinate at the bottom of the image
Output: two lists of inner lane segments
"""

def split_left_right_lines(segments, image_width, min_slope, img_height, lane_band_tolerance):

    left_segments = []
    right_segments = []

    center_x = image_width / 2

    for seg in segments.astype(np.int32):
        x1, y1, x2, y2 = seg

        if x2 == x1:  # Avoid division by zero for vertical lines
            continue

        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2

        if abs(slope) < min_slope:  # Filter out lines that are too horizontal.
            continue

        x_at_bottom = x1 + (img_height - y1) / slope

        if slope < 0 and mid_x < center_x:  # The coordinates of cv are not the same as the math, the left lane has a negative slope. The mid_x is used to determine if the line is on the left or right side of the image.
            left_segments.append({"seg":(x1, y1, x2, y2), "x_at_bottom": x_at_bottom})
        elif slope > 0 and mid_x > center_x:  # The right lane has a positive slope
            right_segments.append({"seg":(x1, y1, x2, y2), "x_at_bottom": x_at_bottom})

    left_segments.sort(key=lambda x: x["x_at_bottom"], reverse=True)  # Sort the left segments in descending order of x_at_bottom
    right_segments.sort(key=lambda x: x["x_at_bottom"])  # Sort the right segments in ascending order of x_at_bottom


    innermost_left_x = left_segments[0]["x_at_bottom"]  # The innermost left lane is the one with the largest x_at_bottom
    inner_left = [l["seg"] for l in left_segments
                    if l["x_at_bottom"] >= (innermost_left_x - lane_band_tolerance)
                    # and l["x_at_bottom"] > center_x * ROI]  # Use lane_band_tolerance to replace the hard threshold of ROI
                   ]

    innermost_right_x = right_segments[0]["x_at_bottom"]
    inner_right = [l["seg"] for l in right_segments
                    if l["x_at_bottom"] <= (innermost_right_x + lane_band_tolerance)
                    # and l["x_at_bottom"] < center_x * (2 - ROI)]
                   ]

    return inner_left, inner_right  # Return type is list of dict

# """
# Lane classification: method 1
# - find the lane seed by selecting the line segment that extends to the bottom of the image and has a certain length.
# - bad with generalization
# """
# def get_best_seed(segments, is_left, image_height, min_length= 30):

#     candidate_seeds = []
#     bottom_threshold = image_height * 0.5

#     for seg in segments:
#         x1, y1, x2, y2 = seg
#         length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

#         if max(y1, y2) > bottom_threshold and length > min_length and x1 != x2:
#             slope = (y2 - y1) / (x2 - x1)
#             x_bottom = x1 + (image_height - y1) / slope  # Extend the line to the bottom of the image
#             candidate_seeds.append({'seg': seg, 'x_bottom': x_bottom})
#     selector = max if is_left else min
#     return selector(candidate_seeds, key= lambda cand: cand['x_bottom'])['seg']

# """
# Lane classification: method 2
# - Cluster the line segments into two groups (left and right) using k-means clustering based on their slope and mid-point y-coordinate.
# - Can't solve the problem of no lane condition in the plane where I'm
# - Can't solve the problem of the curve lane on the slope plane
# """
# def cluster_lines_kmeans(segments, n_clusters=2):

#     features = []
#     for x1, y1, x2, y2 in segments:
#         slope = (y2 - y1) / (x2 - x1)
#         mid_y = (y1 + y2) / 2
#         features.append([slope, mid_y])  # Select slope and mid_y as features for k-means clustering.

#     features = np.array(features)
#     features = StandardScaler().fit_transform(features)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(features)

#     clusters = []
#     for i in range(n_clusters):
#         cluster = [segments[j] for j in range(len(segments)) if labels[j] == i]
#         clusters.append(cluster)

#     return clusters

# def cluster_left_right(left_segments, right_segments, n_clusters=2):
#     left_clusters  = cluster_lines_kmeans(left_segments,  n_clusters)
#     right_clusters = cluster_lines_kmeans(right_segments, n_clusters)

#     return left_clusters, right_clusters
