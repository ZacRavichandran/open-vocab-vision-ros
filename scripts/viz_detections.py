#!/usr/bin/env python

import rospy
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection2D
from visualization_msgs.msg import Marker


class DetectionVizNode:

    RED_COLOR = ColorRGBA(r=0.75, a=1)
    GREEN_COLOR = ColorRGBA(g=0.75, a=1)
    BLUE_COLOR = ColorRGBA(b=0.75, a=1)
    WHITE_COLOR = ColorRGBA(r=0.75, g=0.75, b=0.75, a=1)

    def __init__(self) -> None:
        detection_topic = rospy.get_param("~/detections", "/yolo_ros/detections")

        self.detection_sub = rospy.Subscriber(
            detection_topic, Detection2D, self.detection_cbk
        )

        self.marker_pub = rospy.Publisher("~detection_viz", Marker, queue_size=10)
        # hack to not interfere with tracks
        self.count = 1000
        self.max_msgs = 2000

    def track_cbk(self, track_msg: Detection2D) -> None:
        pass

    def detection_cbk(self, detection_msg: Detection2D) -> None:
        # assume each message has one object :w

        marker_msg = Marker()
        marker_msg.id = self.count
        self.count += 1
        marker_msg.header = detection_msg.header
        marker_msg.pose.position = detection_msg.results[0].pose.pose.position
        marker_msg.pose.orientation = detection_msg.results[0].pose.pose.orientation

        marker_msg.color = self.WHITE_COLOR
        marker_msg.scale.x = 0.25
        marker_msg.scale.y = 0.25
        marker_msg.scale.z = 0.25
        marker_msg.action = marker_msg.ADD
        marker_msg.type = marker_msg.SPHERE

        self.marker_pub.publish(marker_msg)

        if self.count > self.max_msgs:
            self.count = 1000


if __name__ == "__main__":
    rospy.init_node("detection_viz_node")
    viz = DetectionVizNode()
    rospy.spin()
