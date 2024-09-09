# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter


INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric,          # 用什麼方式計算cost
        max_distance,             # cost的閾值
        tracks,
        detections,
        track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)  # 匈牙利算法的實現
    indices = np.hstack([indices[0].reshape(((indices[0].shape[0]), 1)), indices[1].reshape(((indices[0].shape[0]),1))])
    # indices
    # 第幾個target對應到第幾個detection的索引

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)  # 沒有配對到的detection

    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)  # 沒有配對到的tracker

    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric,
        max_distance,  # cost 的 threshold
        cascade_depth, # life time
        tracks,
        detections,
        track_indices=None,
        detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    # print("inside begin")
    # print(track_indices)
    # print(detection_indices)
    # print("inside end")
    unmatched_detections = detection_indices  # detection obj
    matches = []
    for level in range(cascade_depth):  # 0~30
        if len(unmatched_detections) == 0:  # No detections left
            break

        # check track lifetime
        # life time filter
        # track_indices => track_indices_l
        # 最小的 time_since_update=1
        # 按照不同的level關聯
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue
        
        # 匈牙利分層匹配
        # matches, unmatched_tracks, unmatched_detections
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric,
                max_distance,
                tracks,
                detections,
                track_indices_l,
                unmatched_detections)
        matches += matches_l
    
    # print(set(track_indices))
    # print(set(k for k, _ in matches))
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    # 第一round trackers==[] 時
    # matches = []
    # unmatched_tracks = []
    # unmatched_detections = detection_indices
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf,
        cost_matrix,  # 特徵距離，距離越長cost越大
        tracks,
        detections,
        track_indices,
        detection_indices,
        gated_cost=INFTY_COST,  # 100000
        only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4  # 用幾個狀態
    gating_threshold = kalman_filter.chi2inv95[gating_dim]  # 9.4877
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices]) # 所有的detection的xyah
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        # 計算track以及所有測量值
        # 估計值以及測量值的馬氏距離平方
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position) 
        # 判斷馬氏距離要小於gating threshold
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        # cost_matrix[target, :] = gating_distance # 直接用馬氏距離當作 cost function
    return cost_matrix
