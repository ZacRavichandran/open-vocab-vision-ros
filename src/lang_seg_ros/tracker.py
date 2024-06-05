from collections import defaultdict

import numpy as np

import numpy as np

class Hypothesis:
    """hypothesis for single object"""

    def __init__(
        self,
        class_id: int,
        pose: np.ndarray,
        time: float,
        score: float,
        idx: int,
        frame: str,
    ) -> None:
        self.n_detections = 0
        self.class_id = class_id
        self.pose = pose
        self.time = time
        self.score = score
        self.poses = [pose]
        self.idx = idx
        self.frame = frame
        self.velocity = np.zeros(3)
        self.history_weight = 0.95
        print(f"\n\nadding hypothesis: {self.idx}\n\n")

    def update_pose(self, incoming_pose: np.ndarray) -> np.ndarray:
        return (
            1 - self.history_weight
        ) * incoming_pose + self.history_weight * self.pose

    def update_velocity(self, incoming_pose: np.ndarray) -> np.ndarray:
        return (1 - self.history_weight) * (
            incoming_pose - self.pose
        ) + self.history_weight * self.velocity

    def update(self, time: float, pose: np.ndarray, score: float) -> None:
        self.poses.append(pose)
        self.pose = self.update_pose(pose)
        self.velocity = self.update_velocity(pose)
        self.time = time
        self.score = score
        self.n_detections += 1

    def compute_vel(self, history=25):
        pose_history = np.array(self.poses)[-history:]
        filter = np.ones(5) / 5
        filtered_pose = np.stack(
            [np.convolve(pose_history[:, a], filter) for a in range(3)]
        ).T
        filtered_pose = filtered_pose[4:-4]

        cov = np.cov(filtered_pose.T)
        avg_vel = (filtered_pose[-1:] - filtered_pose[1:]).mean(0)
        return avg_vel, cov


class HypothesisSet:
    def __init__(self, dist_threshold: float = 1) -> None:
        # TODO make efficient
        self.hypotheses = []
        self.dist_threshold = dist_threshold
        self.depth_threshold = 7.5

    def add_hypothesis(
        self, time: float, class_id: int, score: float, pose: np.ndarray
    ) -> None:
        pass

    def add_detection(
        self,
        time: float,
        class_id: int,
        score: float,
        pose: np.ndarray,
        n_hypothesis: int,
        frame: str,
    ) -> bool:
        # find best fit pose
        added_hypothesis = False
        min_hypothesis_dist = np.inf
        min_idx = -1
        for idx, hypothesis in enumerate(self.hypotheses):
            dist = np.linalg.norm(hypothesis.pose - pose)
            if dist < min_hypothesis_dist:
                min_hypothesis_dist = dist
                min_idx = idx

        if min_hypothesis_dist < self.dist_threshold:
            self.hypotheses[min_idx].update(time=time, pose=pose, score=score)

        else:
            print(f"min distance: {min_hypothesis_dist:0.2f}. adding new track")
            self.hypotheses.append(
                Hypothesis(
                    class_id=class_id,
                    pose=pose,
                    time=time,
                    score=score,
                    idx=n_hypothesis,
                    frame=frame,
                )
            )
            added_hypothesis = True
        return added_hypothesis

    def print_status(self):
        print(f"hypothesis: ")
        for hypothesis in self.hypotheses:
            print_pose = ", ".join([f"{v:0.2f}" for v in hypothesis.pose])
            print_avg_vel = ", ".join([f"{v:0.2f}" for v in hypothesis.velocity])

            vel, cov = hypothesis.compute_vel()

            print_vel = ", ".join([f"{v:0.2f}" for v in vel])
            print_cov = ", ".join([f"{v:0.2f}" for v in cov[np.diag_indices(3)]])
            det_cov = np.linalg.det(cov)
            print(
                f"\tidx: {hypothesis.idx}, class: {hypothesis.class_id}, pose: ({print_pose}), avg vel: ({print_avg_vel})\n"
                f"\t\t velocity: ({print_vel}), cov: ({print_cov}), det: {det_cov:0.2f}, n_dets: {hypothesis.n_detections}"
            )

        print("---")


class Tracker:
    def __init__(self, distance_threshold: float = 1) -> None:
        class _HypothesisSet(HypothesisSet):
            def __init__(self) -> None:
                super().__init__(distance_threshold)

        self.hypotheses = defaultdict(_HypothesisSet)
        self.n_track_thresh = 25

        # round about for now
        self.hypothesis_idx = 0

    def add_detection(
        self, time: float, class_id: int, score: float, pose: np.ndarray, frame: str
    ) -> None:
        added = self.hypotheses[class_id].add_detection(
            time=time,
            class_id=class_id,
            score=score,
            pose=pose,
            frame=frame,
            n_hypothesis=self.hypothesis_idx,
        )
        if added:
            self.hypothesis_idx += 1

    def get_tracks(self):
        tracks = []
        # sets are organized by class. go through each class set
        for hypothesis_set in self.hypotheses.values():
            # check class hypothesis and check if mature
            for hypothesis in hypothesis_set.hypotheses:
                if hypothesis.n_detections > self.n_track_thresh:
                    tracks.append(hypothesis)

        return tracks

    def print_status(self):
        for hypothesis_set in self.hypotheses.values():
            hypothesis_set.print_status()

