#!/usr/bin/env python
from collections import defaultdict

import numpy as np
import rospy
from geometry_msgs.msg import Point
from lang_seg_ros.tracker import Tracker
from lang_seg_ros.viz_utils import create_marker_msg
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import Detection2D
from visualization_msgs.msg import Marker


class TrackerNode:
    """Multi-hypothesis tracker"""

    def __init__(self) -> None:
        detection_topic = rospy.get_param("~detections", "/yolo_ros/detections")
        track_topic = rospy.get_param("~tracks", "~tracks")
        distance_threshold = rospy.get_param("~distance_threshold", 1)

        self.tracker = Tracker(distance_threshold=distance_threshold)

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

            if self.labels != "":
                text_msg = create_marker_msg(
                    id=2 * track.idx + 1,
                    header=header,
                    color=ColorRGBA(r=0.9, g=0.9, b=0.9, a=1),
                    scale=0.5,
                    position=[track.pose[0], track.pose[1], track.pose[2] + 0.5],
                )
                text_msg.type = marker_msg.TEXT_VIEW_FACING
                text_msg.text = self.labels[track.class_id]
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
