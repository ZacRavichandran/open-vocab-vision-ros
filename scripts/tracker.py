#!/usr/bin/env python
from collections import defaultdict

import numpy as np
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection2D
from visualization_msgs.msg import Marker


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
    def __init__(self) -> None:
        # TODO make efficient
        self.hypotheses = []
        self.dist_threshold = 1
        self.depth_threshold = 10

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
        # don't consider if distance is too far
        # TODO hack to assume map frame is dist from sensor
        if np.linalg.norm(pose) > self.depth_threshold:
            return False

        if np.linalg.norm(pose) > 5:
            print(f"detection is far: {np.linalg.norm(pose)}")

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
    def __init__(self) -> None:
        self.hypotheses = defaultdict(HypothesisSet)
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


class TrackerNode:
    """Multi-hypothesis tracker"""

    def __init__(self) -> None:
        detection_topic = rospy.get_param("~detections", "/yolo_ros/detections")
        track_topic = rospy.get_param("~tracks", "~tracks")

        self.tracker = Tracker()

        self.detection_sub = rospy.Subscriber(
            detection_topic, Detection2D, self.detection_cbk
        )
        self.track_pub = rospy.Publisher(track_topic, Detection2D, queue_size=2)

        self.track_viz = rospy.Publisher("~viz", Marker, queue_size=1)

        self.labels = rospy.get_param("~labels", "")

        if self.labels != "":
            self.labels = self.labels.split(",")

    def get_marker_msg(self, idx, frame, time, position):
        marker_msg = Marker()
        marker_msg.id = idx
        marker_msg.header.frame_id = frame
        marker_msg.header.stamp.secs = np.int32(time // 1)
        marker_msg.header.stamp.nsecs = np.int32(time % 1 * 1e9 // 1)
        marker_msg.pose.position = Point(x=position[0], y=position[1], z=position[2])
        marker_msg.pose.orientation.w = 1

        marker_msg.color = ColorRGBA(r=1, g=0.75, b=0, a=1)
        marker_msg.scale.x = 0.5
        marker_msg.scale.y = 0.5
        marker_msg.scale.z = 0.5
        marker_msg.action = marker_msg.ADD
        marker_msg.type = marker_msg.SPHERE
        return marker_msg

    def pub_tracks(self):
        tracks = self.tracker.get_tracks()

        # lots of duplicate code here
        for track in tracks:
            marker_msg = self.get_marker_msg(
                idx=2 * track.idx,
                frame=track.frame,
                time=track.time,
                position=track.pose,
            )
            self.track_viz.publish(marker_msg)

            if self.labels != "":
                text_msg = self.get_marker_msg(
                    idx=2 * track.idx + 1,
                    frame=track.frame,
                    time=track.time,
                    position=track.pose + np.array([0, 0, 0.5]),
                )
                text_msg.type = marker_msg.TEXT_VIEW_FACING
                text_msg.text = self.labels[track.class_id]
                text_msg.color = ColorRGBA(r=0.9, g=0.9, b=0.9, a=1)
                self.track_viz.publish(text_msg)

        # also need to publish tracks as detection2d msgs

    def detection_cbk(self, detection_msg: Detection2D) -> None:

        pose = detection_msg.results[0].pose.pose.position
        pose = np.array([pose.x, pose.y, pose.z])
        class_id = detection_msg.results[0].id
        score = detection_msg.results[0].score
        time_s = (
            detection_msg.header.stamp.secs + detection_msg.header.stamp.nsecs / 1e9
        )
        self.tracker.add_detection(
            time=time_s,
            class_id=class_id,
            score=score,
            pose=pose,
            frame=detection_msg.header.frame_id,
        )
        # self.tracker.print_status()

        self.pub_tracks()


if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()
