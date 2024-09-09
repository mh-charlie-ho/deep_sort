# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance  # 0.7
        self.max_age = max_age  # 最大留存30幀
        self.n_init = n_init  # 3幀以內都要關聯到(追蹤到)不然就刪除

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            # tracker 本身的預測
            # 均值以及共變異數矩陣的狀態轉移
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # 第一次 [][][all detection]
        # 由feature cost以及iou cost
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        print("check point")
        print(matches)  # [(track_idx, detection_idx), (), (), ...]
        print(unmatched_tracks)
        print(unmatched_detections)
        # input()

        # Update track set.
        # self.hits += 1
        # self.time_since_update = 0
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # 沒有被關聯到30frame(純預測連續30個frame)就刪掉
        # 暫定狀態的就刪掉(連續前3個frame要關聯到)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 沒有被偵測到的detection被初始化
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 只留下存活的
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        print("check point b")
        print(active_targets)
        
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # list len 3
            targets += [track.track_id for _ in track.features]
            track.features = []
        
        # print(features)
        # print(targets)
        # print(active_targets)

        # 更新裡面的feature
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        input()

    def _match(self, detections):

        def gated_metric(
                tracks,                # 所有的 tracker 
                dets,                  # 這一個 frame 的 測量值 
                track_indices,         # confirmed_tracks 對應的 id
                detection_indices):    # 所有的 detection 對應的 id

            # 當前偵測到的detection的特徵
            features = np.array([dets[i].feature for i in detection_indices])  # 這一行要替換掉 ================================
            # confirmed_tracks id
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)  # 計算特徵的距離  # 不需要這一行 因為不用計算特徵的距離 ======================
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # first init
        print(self.metric.matching_threshold)  # 0.2
        print(self.max_age)                    # 30
        print(self.tracks)                     # []
        print(len(detections))                 # 這一幀有9個detection
        print(confirmed_tracks)                # []
        print(unconfirmed_tracks)              # []
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks)
        print("==")
        # 這三個輸出都是索引
        print(matches_a)             # []
        print(unmatched_tracks_a)    # []
        print(unmatched_detections)  # [0,1,2,3,4,5,6,7,8] all detection
        print("heyhey")
        # input()

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # [] + []
        # 只要沒關聯到的新track以及上一個frame有關連到的，遺失一個frame而已
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        # []
        # 遺失幾個frame了
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        
        # 第一次輸出 [] [] []
        # [(track_idx, detection_idx),(),(),...], [], []
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_distance,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,          # 開頭要幾幀內都成功才初始化
                self.max_age,         # 最多維持幾幀
                detection.feature))   # feature塞進去(可以沒有feature)
        self._next_id += 1  # 持續增長，一次的檢測只會出現唯一的track id
