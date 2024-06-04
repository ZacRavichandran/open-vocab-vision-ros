#!/usr/bin/env python
import queue
import time
from typing import Tuple, Union

import cv2
import cv_bridge
import numpy as np
import pyrealsense2 as rs2
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Quaternion
from rosbags.typesys.types import sensor_msgs__msg__CompressedImage as CompressedImgMsg
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMsg
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

try:
    from ultralytics import YOLO
except ImportError:
    raise ValueError("must install ultralytics")


def decode_img_msg(msg: Union[ImageMsg, CompressedImgMsg]) -> np.ndarray:
    """Decode ROS image message.

    This implements functionality of cv_bridge (was having issues with some
    dependencies. This was the simplest solution).

    Parameters
    ----------
    msg : Union[ImageMsg, CompressedImgMsg]
        Incoming ROS image message.

    Returns
    -------
    np.ndarray
        Decoded image as numpy array.

    Raises
    ------
    ValueError
        Raises error if image encoding is not implemented.
    """
    if "Compressed" in str(type(msg)):  # better way?
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        print(img.shape)
    else:
        if msg.encoding == "rgb8":
            img = np.copy(
                np.ndarray(
                    shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data
                )
            )
        elif msg.encoding == "16UC1":
            # with open("/home/zac/test_img.txt", "wb") as f:
            #     f.write(msg.data)
            dtype = np.dtype("uint16")
            dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
            img = np.ndarray(
                shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data
            )

        else:
            raise ValueError(f"{msg.encoding} not supported")
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
        # read parameters
        color_sub_topic = rospy.get_param("~input", "/camera/color/image_raw")
        depth_info_sub_topic = rospy.get_param(
            "~depth_aligned_info", "/camera/aligned_depth_to_color/camera_info"
        )
        depth_sub_topic = rospy.get_param(
            "~depth_aligned_img", "/camera/aligned_depth_to_color/image_raw"
        )
        pub_topic = rospy.get_param("~output", "~pred")
        weights = rospy.get_param(
            "~weights",
        )
        self.drop_old_msg = rospy.get_param("~drop", True)
        self.debug = rospy.get_param("~debug", True)
        self.target_frame = rospy.get_param("~target_frame", "map")
        self.labels = rospy.get_param("~labels", "")

        # setup class members
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

        self.intrinsics = None
        self.last_depth = None

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # setup subscribers
        self.img_sub = rospy.Subscriber(
            color_sub_topic, Image, self.img_callback, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(depth_sub_topic, Image, self.depth_callback)
        self.depth_info_sub = rospy.Subscriber(
            depth_info_sub_topic,
            CameraInfo,
            self.depth_info_callback,
        )

        # setup publishers
        self.img_pub = rospy.Publisher(pub_topic, Image, queue_size=1)
        self.debug_pub = rospy.Publisher("~debug", Image, queue_size=1)
        self.detection_pub = rospy.Publisher("~detections", Detection2D, queue_size=1)

        # ROS will drop incoming messages if subscriber is full. It would
        # be preferable to drop old messages, so we'll use a queue as
        # an intermediate container

        # TODO still an issue where queue is read too fast
        while True:
            if not self.img_queue.empty():
                if self.img_queue.qsize():
                    img = self.img_queue.get(block=True)
                    self.detect(img)
                    rospy.sleep(1e-3)

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

    def depth_callback(self, depth_msg: Image) -> None:
        self.last_depth = depth_msg

    def depth_info_callback(self, camera_info: CameraInfo) -> None:
        # from rs2 / show_center_depth.py
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = camera_info.width
            self.intrinsics.height = camera_info.height
            self.intrinsics.ppx = camera_info.K[2]
            self.intrinsics.ppy = camera_info.K[5]
            self.intrinsics.fx = camera_info.K[0]
            self.intrinsics.fy = camera_info.K[4]
            if camera_info.distortion_model == "plumb_bob":
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif camera_info.distortion_model == "equidistant":
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in camera_info.D]

        except cv_bridge.CvBridgeError as e:
            print(e)
            return

    def detect(self, img_msg: Image) -> None:
        img = decode_img_msg(img_msg)
        pred = self.yolo_infer.predict(img)

        pred_color = pred[0].plot()
        color_msg = self.bridge.cv2_to_imgmsg(pred_color, encoding="passthrough")
        color_msg.header = img_msg.header  # TODO do we want this?
        color_msg.encoding = "rgb8"

        self.img_pub.publish(color_msg)

        boxes = pred[0].boxes.xyxy.cpu().numpy()
        classes = pred[0].boxes.cls.cpu().numpy().astype(np.int16)
        confidences = pred[0].boxes.conf.cpu().numpy()
        assert (
            img_msg.header.frame_id == "camera_color_optical_frame"
        ), f" img frame: {img_msg.header.frame_id}"

        for box, class_id, conf in zip(boxes, classes, confidences):
            x, y, z = self.deproject_detections(
                box[::2].mean(),
                box[1::2].mean(),
                w=box[2] - box[0],
                h=box[3] - box[1],
                time=img_msg.header.stamp,
            )
            self.publish_detection_msg(
                class_id=class_id,
                confidence=conf,
                pose=(x, y, z),
                header=img_msg.header,
            )

    def publish_detection_msg(
        self,
        class_id: int,
        confidence: float,
        pose: Tuple[float, float, float],
        header: Header,
    ) -> None:
        object_msgs = []
        object_msg = ObjectHypothesisWithPose()
        object_msg.id = class_id
        object_msg.score = confidence
        object_msg.pose.pose.position = Point(x=pose[0], y=pose[1], z=pose[2])
        object_msg.pose.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        object_msgs.append(object_msg)
        detection_msg = Detection2D()
        detection_msg.header = header
        detection_msg.header.frame_id = self.target_frame  # TODO yes?
        detection_msg.results = object_msgs

        self.detection_pub.publish(detection_msg)

    def deproject_detections(
        self, x: float, y: float, w: float, h: float, time: float
    ) -> Tuple[float, float, float]:
        if self.last_depth == None or self.intrinsics == None:
            return

        # depth is given in mm. We convert that to meters
        depth_img = decode_img_msg(self.last_depth)
        depth_img = depth_img / 1000

        int_x, int_y = np.int16(x), np.int16(y)

        # take 10% crop around box to reduce noise
        w = np.maximum(w * 0.1, 2).astype(np.int32)
        h = np.maximum(h * 0.1, 2).astype(np.int32)

        depth_point = depth_img[
            int_y - h // 2 : int_y + h // 2, int_x - w // 2 : int_x + w // 2
        ]
        depth_point = depth_point.mean()

        result_camera_coords = rs2.rs2_deproject_pixel_to_point(
            self.intrinsics, (int_x, int_y), depth_point
        )

        result_camera_coords = np.array(result_camera_coords)

        transform_msg = self.tf_buffer.lookup_transform(
            self.target_frame, "camera_color_optical_frame", rospy.Time()
        )

        transform = transform_msg.transform

        rot = Rotation.from_quat(
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ]
        )
        trans = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z]
        )

        result_map = rot.as_matrix() @ result_camera_coords + trans

        x = result_map[0]
        y = result_map[1]
        z = result_map[2]

        return x, y, z



if __name__ == "__main__":
    rospy.init_node("lang_seg_ros")

    infer = LangSegInferRos()
