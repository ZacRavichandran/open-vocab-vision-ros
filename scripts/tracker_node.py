#!/usr/bin/env python
from typing import Sequence

import numpy as np
import rospy
from geometry_msgs.msg import Point
from open_vocab_vision_ros.msg import Track, Detection
from open_vocab_vision_ros.ros_utils import to_track_msg
from open_vocab_vision_ros.tracker import Hypothesis, Tracker
from open_vocab_vision_ros.viz_utils import create_marker_msg
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker


class TrackerNode:
    """Multi-hypothesis tracker"""

    def __init__(self) -> None:
        detection_topic = rospy.get_param("~detections", "/yolo_ros/detections")
        track_topic = rospy.get_param("~tracks", "~tracks")
        distance_threshold = rospy.get_param("~distance_threshold", 1)
        n_track_thresh = rospy.get_param("~n_track_thresh", 25)

        self.tracker = Tracker(
            distance_threshold=distance_threshold, n_track_thresh=n_track_thresh
        )

        self.detection_sub = rospy.Subscriber(
            detection_topic, Detection, self.detection_cbk
        )
        self.track_pub = rospy.Publisher(track_topic, Track, queue_size=10)

        self.track_viz = rospy.Publisher("~viz", Marker, queue_size=1)

        self.labels = rospy.get_param("~labels", "")

        if self.labels != "":
            self.labels = self.labels.split(",")
        else:
            self.labels = []

    def get_marker_msg(
        self, idx: int, frame: str, time: float, position: Sequence[float]
    ) -> Marker:
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

    def pub_tracks(self) -> None:
        tracks = self.tracker.get_tracks()

        for track in tracks:
            self.publish_markers(track)
            self.track_pub.publish(to_track_msg(track))
            # self.tracker.log_status()

    def publish_markers(self, track: Hypothesis) -> None:
        header = Header()
        header.frame_id = track.frame
        header.stamp.secs = np.int32(track.time // 1)
        header.stamp.nsecs = np.int32(track.time % 1 * 1e9 // 1)

        marker_msg = create_marker_msg(
            id=2 * track.idx,
            header=header,
            color=ColorRGBA(r=1, g=0.75, b=0, a=1),
            scale=0.5,
            position=[track.pose[0], track.pose[1], track.pose[2]],
        )
        self.track_viz.publish(marker_msg)

        if len(self.labels):
            text_msg = create_marker_msg(
                id=2 * track.idx + 1,
                header=header,
                color=ColorRGBA(r=0.9, g=0.9, b=0.9, a=1),
                scale=0.5,
                position=[track.pose[0], track.pose[1], track.pose[2] + 0.5],
            )
            text_msg.type = marker_msg.TEXT_VIEW_FACING
            text_msg.text = track.label 
            self.track_viz.publish(text_msg)

    def detection_cbk(self, detection_msg: Detection) -> None:
        assert len(detection_msg.results) == 1, "cannot support >1 atm"
        pose = detection_msg.results[0].pose.pose.position
        pose = np.array([pose.x, pose.y, pose.z])
        class_id = detection_msg.results[0].id
        score = detection_msg.results[0].score
        label = detection_msg.labels[0]
        time_s = (
            detection_msg.header.stamp.secs + detection_msg.header.stamp.nsecs / 1e9
        )
        self.tracker.add_detection(
            time=time_s,
            class_id=class_id,
            score=score,
            pose=pose,
            frame=detection_msg.header.frame_id,
            label=label,
        )
        # self.tracker.log_status()

        self.pub_tracks()


if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()
