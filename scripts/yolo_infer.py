#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np

import time
import cv2
import cv_bridge
import queue

from typing import Union
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMsg
from rosbags.typesys.types import sensor_msgs__msg__CompressedImage as CompressedImgMsg

try:
    from ultralytics import YOLO
except ImportError:
    raise ValueError("must install lseg")


def decode_img_msg(msg: Union[ImageMsg, CompressedImgMsg]) -> np.ndarray:
    if "Compressed" in str(type(msg)):  # better way?
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        print(img.shape)
    else:
        assert msg.encoding == "rgb8", "other values not supported"
        img = np.copy(
            np.ndarray(
                shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data
            )
        )
    return img


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
        sub_topic = rospy.get_param("~input", "/camera/color/image_raw")
        pub_topic = rospy.get_param("~output", "~pred")
        weights = rospy.get_param(
            "~weights",
        )
        self.drop_old_msg = rospy.get_param("~drop", True)
        self.debug = rospy.get_param("~debug", True)
        self.target_frame = rospy.get_param(
            "~target_frame", "camera_color_optical_frame"
        )

        self.labels = rospy.get_param("~labels", "")
        if self.labels != "":
            self.labels = self.labels.split(",")
        else:
            self.labels = self.DEFAULT_LABELS
        rospy.loginfo(f"using labels: {self.labels}")

        self.bridge = cv_bridge.CvBridge()
        self.img_queue = queue.Queue(maxsize=2)

        self.yolo_infer = YOLO(weights)
        self.yolo_infer.set_classes(self.labels)
        rospy.loginfo("yololoaded.")

        self.img_sub = rospy.Subscriber(
            sub_topic, Image, self.img_callback, queue_size=1
        )
        self.img_pub = rospy.Publisher(pub_topic, Image, queue_size=1)
        # self.debug_pub = rospy.Publisher("~debug", Image, queue_size=1)

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

        # doesn't seem to work as well on smaller images
        # resizing = False
        if img.shape[0] < 500:
            img = cv2.resize(img, (0, 0), fx=2, fy=2)
            resizing = True

        print(img.dtype, img.max(), img.min())

        t1 = time.time()
        pred = self.yolo_infer.predict(img)

        # if resizing:
        #     pred = pred[..., ::2, ::2]

        rospy.loginfo(f"lseg inference on ({img.shape}) img: {time.time() - t1:0.3f}s")

        pred_color = pred[0].plot()

        color_msg = self.bridge.cv2_to_imgmsg(pred_color, encoding="passthrough")
        color_msg.header = img_msg.header  # TODO do we want this?
        color_msg.header.frame_id = self.target_frame  # TODO bit of a hack
        color_msg.encoding = "rgb8"

        self.img_pub.publish(color_msg)

        # if self.debug:
        #     fig, ax, plot = get_result_plot(img, pred_color, patches)
        #     debug_msg = self.bridge.cv2_to_imgmsg(plot, encoding="passthrough")
        #     debug_msg.header = img_msg.header
        #     debug_msg.encoding = "rgb8"
        #     self.debug_pub.publish(debug_msg)


if __name__ == "__main__":
    rospy.init_node("lang_seg_ros")

    infer = LangSegInferRos()
