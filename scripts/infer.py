#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np

import time
import cv2
import cv_bridge
import queue

from lseg.inference import LangSegInference, decode_img_msg


class LangSegInferRos:
    DEFAULT_LABELS = [
        "road",
        "car",
        "sky",
        "building",
        "person",
        "fence",
        "grass",
        "tree",
        "bush",
        "dumpster",
        "wall",
        "truck",
    ]

    def __init__(self):
        sub_topic = rospy.get_param("~input")
        pub_topic = rospy.get_param("~output")
        weights = rospy.get_param("~weights")
        self.drop_old_msg = rospy.get_param("~drop", True)

        self.bridge = cv_bridge.CvBridge()
        self.img_queue = queue.Queue(maxsize=2)

        self.lseg_infer = LangSegInference(labels=self.DEFAULT_LABELS, weights=weights)
        rospy.loginfo("lang-seg loaded.")

        self.img_sub = rospy.Subscriber(
            sub_topic, Image, self.img_callback, queue_size=1
        )
        self.img_pub = rospy.Publisher(pub_topic, Image, queue_size=1)

        # ROS will drop incoming messages if subscriber is full. It would
        # be prefereable to drop old messages, so we'll use a queue as
        # an intermediate container
        while True:
            if not self.img_queue.empty():
                img = self.img_queue.get(block=False)
                self.segment(img)
                rospy.sleep(1e-3)

        rospy.spin()  # can remove

    def img_callback(self, img_msg: Image) -> None:
        """If `self.drop_old_msg` is true, empty the queue before
        placing image so that we segment the most recent one / don't lag.

        Parameters
        ----------
        img_msg : Image
            Input image
        """
        while self.drop_old_msg and not self.img_queue.empty():
            self.img_queue.get(block=False)

        self.img_queue.put(img_msg)

    def segment(self, img_msg: Image) -> None:
        img = decode_img_msg(img_msg)
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        t1 = time.time()
        pred = self.lseg_infer.infer(img, in_color=False)
        rospy.loginfo(f"lseg inference on ({img.shape}) img: {time.time() - t1:0.3f}s")

        pred_color, patches = self.lseg_infer.get_class_colored_img(pred)
        pred_color = np.array(pred_color)[..., :3]

        color_msg = self.bridge.cv2_to_imgmsg(pred_color, encoding="passthrough")
        color_msg.header = img_msg.header  # TODO do we want this?
        color_msg.encoding = "rgb8"

        self.img_pub.publish(color_msg)


if __name__ == "__main__":
    rospy.init_node("lang_seg_ros")

    infer = LangSegInferRos()
rospy.loginfo("lang-seg loaded.")
