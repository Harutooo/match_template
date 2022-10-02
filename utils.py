"""Various, mostly statistical, utility functions.
"""

from itertools import groupby

import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
from scipy.signal import argrelmin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity


def timeify(time):
    """Format a time in seconds to a minutes/seconds timestamp."""
    time = float(time)
    mins, secs = time // 60, time % 60
    return f"{mins:.0f}:{secs:05.2f}"

# clusters = get_clusters(good_points, max_distance=self.max_distance)
def get_clusters(pts, key=lambda x: x, max_clusters=None, max_distance=14):
    """Run DBSCAN on the `pts`, applying `key` first if necessary,
    post-process the results into a list of lists, and return it,
    taking only the largest `max_clusters` clusters.
    """
    if pts:
        kpts = [key(pt) for pt in pts]

        clustering = DBSCAN(eps=max_distance, min_samples=1).fit(kpts)

        '''
        clustering.labels_ : 
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 1
        1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        2 2 2 2 2 2 2 2 2 2 4 4 4 4 4 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
        4 4 4 4 4 4 4 4 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 4 4 2 2 2 2 2 2 2 2
        2 2 2 2 2 2 2 2 2 2 2 2]
        '''
        # Post-processing.
        labeled_pts = list(zip(kpts, clustering.labels_))
        labeled_pts = sorted(labeled_pts, key=lambda p: p[1])

        '''
        labeled_pts : 
        [((18, 739), 0), ((18, 740), 0), ((18, 741), 0), ((18, 748), 0), ((19, 741), 0), ((19, 742), 0), 
        ((19, 743), 0), ((19, 744), 0), ((19, 745), 0), ((20, 742), 0), ((20, 743), 0), ((20, 744), 0), 
        ((21, 743), 0), ((22, 791), 0), ((22, 793), 0), ((22, 794), 0), ((22, 795), 0), ((22, 796), 0), 
        ((23, 786), 0), ((23, 788), 0), ((23, 789), 0), ((24, 729), 0), ((24, 730), 0), ((24, 790), 0), 
        ((24, 791), 0), ((24, 792), 0), ((24, 793), 0), ((24, 794), 0), ((49, 697), 0), ((49, 698), 0), 
        ((49, 706), 0), ((59, 692), 0), ((41, 597), 1), ((41, 598), 1), ((67, 553), 1), ((67, 554), 1), 
        ((67, 555), 1), ((67, 562), 1), ((67, 563), 1), ((98, 570), 1), ((99, 570), 1), ((78, 137), 2), 
        ((78, 138), 2), ((78, 139), 2), ((79, 22), 2), ((79, 23), 2), ((79, 24), 2), ((79, 137), 2), 
        ((79, 138), 2), ((79, 139), 2), ((79, 140), 2), ((80, 22), 2), ((83, 39), 2), ((83, 40), 2), 
        ((83, 41), 2), ((85, 8), 2), ((119, 101), 2), ((119, 102), 2), ((119, 103), 2), ((119, 104), 2), 
        ((119, 105), 2), ((119, 106), 2), ((119, 107), 2), ((119, 108), 2), ((119, 109), 2), ((119, 110), 2), 
        ((119, 111), 2), ((119, 112), 2), ((120, 106), 2), ((120, 107), 2), ((120, 108), 2), ((120, 109), 2), 
        ((120, 110), 2), ((120, 111), 2), ((121, 108), 2), ((129, 38), 2), ((130, 39), 2), ((145, 113), 2), 
        ((147, 92), 2), ((147, 93), 2), ((147, 94), 2), ((147, 95), 2), ((147, 96), 2), ((147, 97), 2), 
        ((147, 98), 2), ((147, 99), 2), ((147, 100), 2), ((147, 101), 2), ((147, 102), 2), ((147, 103), 2), 
        ((148, 98), 2), ((148, 99), 2), ((148, 100), 2), ((148, 101), 2), ((149, 125), 2), ((179, 29), 2), 
        ((179, 30), 2), ((185, 85), 2), ((193, 187), 2), ((193, 188), 2), ((193, 195), 2), ((193, 196), 2), 
        ((196, 525), 2), ((196, 529), 2), ((196, 530), 2), ((196, 531), 2), ((196, 532), 2), ((196, 533), 2), 
        ((196, 534), 2), ((196, 535), 2), ((197, 445), 2), ((197, 446), 2), ((197, 530), 2), ((197, 531), 2), 
        ((197, 532), 2), ((197, 533), 2), ((198, 449), 2), ((198, 450), 2), ((198, 451), 2), ((199, 448), 2), 
        ((218, 411), 2), ((218, 412), 2), ((218, 415), 2), ((218, 419), 2), ((219, 403), 2), ((219, 404), 2), 
        ((219, 405), 2), ((219, 406), 2), ((219, 409), 2), ((219, 411), 2), ((219, 412), 2), ((219, 413), 2), 
        ((219, 414), 2), ((220, 411), 2), ((223, 317), 2), ((223, 318), 2), ((223, 325), 2), ((223, 326), 2), 
        ((224, 180), 2), ((224, 181), 2), ((224, 182), 2), ((224, 183), 2), ((224, 184), 2), ((224, 185), 2), 
        ((224, 186), 2), ((224, 191), 2), ((224, 192), 2), ((224, 193), 2), ((224, 194), 2), ((224, 196), 2), 
        ((224, 199), 2), ((224, 200), 2), ((224, 201), 2), ((224, 202), 2), ((225, 173), 2), ((225, 174), 2), 
        ((225, 176), 2), ((225, 177), 2), ((225, 178), 2), ((225, 179), 2), ((225, 180), 2), ((225, 181), 2), 
        ((225, 182), 2), ((225, 183), 2), ((225, 184), 2), ((225, 185), 2), ((225, 186), 2), ((225, 187), 2), 
        ((225, 188), 2), ((225, 189), 2), ((225, 190), 2), ((225, 191), 2), ((225, 192), 2), ((225, 193), 2), 
        ((225, 194), 2), ((225, 195), 2), ((225, 196), 2), ((225, 197), 2), ((226, 168), 2), ((226, 169), 2), 
        ((226, 174), 2), ((226, 175), 2), ((226, 176), 2), ((226, 177), 2), ((226, 178), 2), ((226, 179), 2), 
        ((226, 180), 2), ((226, 181), 2), ((226, 184), 2), ((226, 188), 2), ((227, 171), 2), ((227, 172), 2), 
        ((227, 173), 2), ((227, 174), 2), ((227, 175), 2), ((228, 172), 2), ((228, 173), 2), ((229, 170), 2), 
        ((229, 171), 2), ((229, 441), 2), ((237, 309), 2), ((237, 310), 2), ((238, 314), 2), ((239, 405), 2), 
        ((239, 406), 2), ((239, 407), 2), ((240, 390), 2), ((240, 391), 2), ((240, 392), 2), ((240, 393), 2), 
        ((240, 401), 2), ((241, 389), 2), ((241, 390), 2), ((241, 391), 2), ((241, 392), 2), ((241, 393), 2), 
        ((241, 394), 2), ((241, 395), 2), ((241, 396), 2), ((249, 21), 2), ((252, 273), 2), ((252, 274), 2), 
        ((252, 275), 2), ((252, 276), 2), ((252, 282), 2), ((252, 283), 2), ((253, 263), 2), ((253, 264), 2), 
        ((253, 265), 2), ((253, 269), 2), ((253, 270), 2), ((253, 271), 2), ((253, 272), 2), ((253, 275), 2), 
        ((253, 276), 2), ((253, 277), 2), ((253, 278), 2), ((253, 279), 2), ((253, 280), 2), ((154, 649), 3), 
        ((215, 764), 4), ((215, 765), 4), ((215, 772), 4), ((215, 773), 4), ((215, 774), 4), ((229, 741), 4), 
        ((229, 742), 4), ((229, 743), 4), ((229, 744), 4), ((229, 749), 4), ((229, 750), 4), ((229, 751), 4), 
        ((229, 752), 4), ((229, 753), 4), ((230, 735), 4), ((230, 736), 4), ((230, 737), 4), ((230, 738), 4), 
        ((230, 741), 4), ((230, 742), 4), ((230, 743), 4), ((230, 744), 4), ((230, 745), 4), ((230, 746), 4), 
        ((230, 747), 4), ((230, 748), 4), ((231, 745), 4), ((231, 746), 4), ((232, 746), 4), ((243, 682), 4), 
        ((243, 683), 4)]
        '''

        clusters = [
            list(g) for _, g in groupby(labeled_pts, key=lambda p: p[1])
        ]
        '''
        每个类别单独[]起来
        [[((18, 739), 0), ((18, 740), 0), ((18, 741), 0), ((18, 748), 0), ((19, 741), 0), ((19, 742), 0), 
        ((19, 743), 0), ((19, 744), 0), ((19, 745), 0), ((20, 742), 0), ((20, 743), 0), ((20, 744), 0)], 
        ...
        [((41, 597), 1), ((41, 598), 1), ((67, 553), 1), ((67, 554), 1), 
        ((67, 555), 1), ((67, 562), 1), ((67, 563), 1), ((98, 570), 1), ((99, 570), 1)], 
        
        [((78, 138), 2), ((78, 139), 2), ((79, 22), 2), ((79, 23), 2), ((79, 24), 2), ((79, 137), 2), 
        ...
        ((253, 265), 2), ((253, 269), 2), ((253, 270), 2), ((253, 271), 2), ((253, 272), 2), ((253, 275), 2), 
        ((253, 276), 2), ((253, 277), 2), ((253, 278), 2), ((253, 279), 2), ((253, 280), 2)], 
        
        [((154, 649), 3)],

        [((215, 764), 4), ((215, 765), 4), ((215, 772), 4), ((215, 773), 4), ((215, 774), 4), ((229, 741), 4), 
        ... 
        ((243, 683), 4)]]

        '''

        # 去掉类别
        clusters = [[p[0] for p in clust] for clust in clusters]
        # print('clusters : ')
        # print(len(clusters))
        clusters = list(sorted(clusters, key=len, reverse=True))

        # print('clusters[:max_clusters]')
        # print(clusters[-1])
        return clusters[:max_clusters]
    return []


def compute_minimum_kernel_density(series):
    """Estimate the value within the range of _series_ that is the furthest
    away from most observations.
    """
    # Trim outliers for robustness.
    p05, p95 = series.quantile((0.05, 0.95))
    samples = np.linspace(p05, p95, num=100)

    # Find the minimum kernel density.
    kde = KernelDensity(kernel="gaussian", bandwidth=0.005)
    kde = kde.fit(np.array(series).reshape(-1, 1))
    estimates = kde.score_samples(samples.reshape(-1, 1))

    rel_mins = argrelmin(estimates)[0]

    def depth(idx):
        return min(
            estimates[idx - 1] - estimates[idx],
            estimates[idx + 1] - estimates[idx],
        )

    deepest_min = max(rel_mins, key=depth)

    return samples[deepest_min]


def scale_to_interval(array, new_min, new_max):
    """Scale the elements of _array_ linearly to lie between
    _new_min_ and _new_max_.
    """
    array_min = min(array.flatten())
    array_max = max(array.flatten())

    # array_01 is scaled between 0 and 1.
    if array_min == array_max:
        array_01 = np.zeros(array.shape)
    else:
        array_01 = (array - array_min) / (array_max - array_min)

    return new_min + (new_max - new_min) * array_01


def overlay_map(frames):
    """Run a skewness-kurtosis filter on a sample of frames and
    edge-detect.
    The areas of the video containing game feed should come back black.
    Areas containing overlay or letterboxes will be visibly white.
    """

    skew_map = scipy.stats.skew(frames, axis=0)
    kurt_map = scipy.stats.kurtosis(frames, axis=0)
    min_map = np.minimum(
        skew_map, kurt_map
    )  # pylint:disable=assignment-from-no-return

    min_map = scale_to_interval(min_map, 0, 255).astype(np.uint8)

    # Blur and edge detect.
    min_map = cv2.blur(min_map, (5, 5))
    edges = cv2.Laplacian(min_map, cv2.CV_8U)

    # Areas that are constant throughout the video (letterboxes) will
    # have 0 skew, 0 kurt, and 0 variance, so the skew-kurt filter
    # will miss them
    sd_map = np.sqrt(np.var(frames, axis=0))
    edges[np.where(sd_map < 0.01)] = 255
    _, edges = cv2.threshold(edges, 7, 255, cv2.THRESH_BINARY)

    return edges


def find_dlt(predicted, locations):
    """Determine the direct linear transformation that moves the percent signs
    to where they should be using OLS (ordinary least squares.)
    Specifically, compute the OLS solution of the following system:
    port_0_x_predicted * scale + shift_x = port_0_x_actual
    port_0_y_predicted * scale + shift_y = port_0_y_actual
    ...
    port_4_x_predicted * scale + shift_x = port_4_x_actual
    port_4_y_predicted * scale + shift_y = port_4_y_actual
    In matrix form Ax = b :
    [ p0x_pred 1 0 ] [ scale   ] = [ p0x_actual ]
    [ p0y_pred 0 1 ] [ shift_x ]   [ p0y_actual ]
    [ ...          ] [ shift_y ]   [ ...]
    [ p4x_pred 1 0 ]               [ p4x_actual ]
    [ p4y_pred 0 1 ]               [ p4x_actual ]
    """

    predicted_mat = []
    for (predicted_y, predicted_x) in predicted:
        predicted_mat.append([predicted_y, 0, 1])
        predicted_mat.append([predicted_x, 1, 0])

    actual_vec = []
    for (actual_y, actual_x) in locations:
        actual_vec.append(actual_y)
        actual_vec.append(actual_x)
    actual_vec = np.array(actual_vec).transpose()

    # TODO Check this thing's robustness
    ols, resid, _, _ = np.linalg.lstsq(predicted_mat, actual_vec, rcond=None)

    scale_factor, shift_x, shift_y = ols
    return (scale_factor, shift_x, shift_y)


def bisect(f, start, end, tolerance):
    # First make sure we have an interval to which bisection is applicable
    # (that is, one on which f(t) changes sign.)
    # Also compute start and end confs.
    start_value = f(start)
    end_value = f(end)

    plus_to_minus = start_value > 0 > end_value
    minus_to_plus = start_value < 0 < end_value

    if not (minus_to_plus or plus_to_minus):
        raise ValueError(f"bisect() got a bad interval [{start}, {end}]")

    while end - start > tolerance:
        middle = (start + end) / 2
        middle_value = f(middle)
        if (0 > middle_value and 0 > start_value) or (
            0 < middle_value and 0 < start_value
        ):
            start = middle
        else:
            end = middle

    return (start + end) / 2

# 仅仅用来调试
if __name__ == '__main__':
    good_points = [
    (18, 739), (18, 740), (18, 741), (18, 748), (19, 741), (19, 742), (19, 743), (19, 744), (19, 745), 
    (20, 742), (20, 743), (20, 744), (21, 743), (22, 791), (22, 793), (22, 794), (22, 795), (22, 796), 
    (23, 786), (23, 788), (23, 789), (24, 729), (24, 730), (24, 790), (24, 791), (24, 792), (24, 793), 
    (24, 794), (41, 597), (41, 598), (49, 697), (49, 698), (49, 706), (59, 692), (67, 553), (67, 554), 
    (67, 555), (67, 562), (67, 563), (78, 137), (78, 138), (78, 139), (79, 22), (79, 23), (79, 24), 
    (79, 137), (79, 138), (79, 139), (79, 140), (80, 22), (83, 39), (83, 40), (83, 41), (85, 8), (98, 570), 
    (99, 570), (119, 101), (119, 102), (119, 103), (119, 104), (119, 105), (119, 106), (119, 107), (119, 108), 
    (119, 109), (119, 110), (119, 111), (119, 112), (120, 106), (120, 107), (120, 108), (120, 109), (120, 110), 
    (120, 111), (121, 108), (129, 38), (130, 39), (145, 113), (147, 92), (147, 93), (147, 94), (147, 95), 
    (147, 96), (147, 97), (147, 98), (147, 99), (147, 100), (147, 101), (147, 102), (147, 103), (148, 98), 
    (148, 99), (148, 100), (148, 101), (149, 125), (154, 649), (179, 29), (179, 30), (185, 85), (193, 187), 
    (193, 188), (193, 195), (193, 196), (196, 525), (196, 529), (196, 530), (196, 531), (196, 532), (196, 533), 
    (196, 534), (196, 535), (197, 445), (197, 446), (197, 530), (197, 531), (197, 532), (197, 533), (198, 449), 
    (198, 450), (198, 451), (199, 448), (215, 764), (215, 765), (215, 772), (215, 773), (215, 774), (218, 411), 
    (218, 412), (218, 415), (218, 419), (219, 403), (219, 404), (219, 405), (219, 406), (219, 409), (219, 411), 
    (219, 412), (219, 413), (219, 414), (220, 411), (223, 317), (223, 318), (223, 325), (223, 326), (224, 180), 
    (224, 181), (224, 182), (224, 183), (224, 184), (224, 185), (224, 186), (224, 191), (224, 192), (224, 193), 
    (224, 194), (224, 196), (224, 199), (224, 200), (224, 201), (224, 202), (225, 173), (225, 174), (225, 176), 
    (225, 177), (225, 178), (225, 179), (225, 180), (225, 181), (225, 182), (225, 183), (225, 184), (225, 185), 
    (225, 186), (225, 187), (225, 188), (225, 189), (225, 190), (225, 191), (225, 192), (225, 193), (225, 194), 
    (225, 195), (225, 196), (225, 197), (226, 168), (226, 169), (226, 174), (226, 175), (226, 176), (226, 177), 
    (226, 178), (226, 179), (226, 180), (226, 181), (226, 184), (226, 188), (227, 171), (227, 172), (227, 173), 
    (227, 174), (227, 175), (228, 172), (228, 173), (229, 170), (229, 171), (229, 441), (229, 741), (229, 742), 
    (229, 743), (229, 744), (229, 749), (229, 750), (229, 751), (229, 752), (229, 753), (230, 735), (230, 736), 
    (230, 737), (230, 738), (230, 741), (230, 742), (230, 743), (230, 744), (230, 745), (230, 746), (230, 747), 
    (230, 748), (231, 745), (231, 746), (232, 746), (237, 309), (237, 310), (238, 314), (239, 405), (239, 406), 
    (239, 407), (240, 390), (240, 391), (240, 392), (240, 393), (240, 401), (241, 389), (241, 390), (241, 391), 
    (241, 392), (241, 393), (241, 394), (241, 395), (241, 396), (243, 682), (243, 683), (249, 21), (252, 273), 
    (252, 274), (252, 275), (252, 276), (252, 282), (252, 283), (253, 263), (253, 264), (253, 265), (253, 269), 
    (253, 270), (253, 271), (253, 272), (253, 275), (253, 276), (253, 277), (253, 278), (253, 279), (253, 280)]
    # print(len(good_points))
    clusters = get_clusters(good_points, max_distance=80)
    # # y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
    # plt.scatter(good_points[:, 0], good_points[:, 1], c=clusters)
    # plt.show()
    # print(clusters)
