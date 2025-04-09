import queue
from typing import Tuple, Union, List

import cv2
import cv_bridge
import numpy as np
import pyrealsense2 as rs2
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Quaternion
from rosbags.typesys.types import sensor_msgs__msg__CompressedImage as CompressedImgMsg
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMsg
from open_vocab_vision_ros.msg import Detection
from open_vocab_vision_ros.srv import (
    SetLabels,
    SetLabelsRequest,
    SetLabelsResponse,
    GetLabels,
    GetLabelsRequest,
    GetLabelsResponse,
)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import ObjectHypothesisWithPose
from visualization_msgs.msg import Marker

from open_vocab_vision_ros.viz_utils import create_marker_msg


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
        rospy.logdebug(f"incoming image shape: {img.shape}")
    else:
        if msg.encoding == "rgb8":
            img = np.copy(
                np.ndarray(
                    shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data
                )
            )
        elif msg.encoding == "bgra8":
            img = np.copy(
                np.ndarray(
                    shape=(msg.height, msg.width, 4), dtype=np.uint8, buffer=msg.data
                )
            )[:, :, :3]
            img = img[..., ::-1]  # swap r and b channels
        elif msg.encoding == "rgba8":
            img = np.copy(
                np.ndarray(
                    shape=(msg.height, msg.width, 4), dtype=np.uint8, buffer=msg.data
                )
            )
            img = img[..., :3]
        elif msg.encoding == "32FC1":
            dtype = np.dtype("float32")
            dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
            img = np.ndarray(
                shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data
            ).copy()
        elif msg.encoding == "16UC1":
            dtype = np.dtype("uint16")
            dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
            img = np.ndarray(
                shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data
            )

        else:
            raise ValueError(f"{msg.encoding} not supported")
    return img


class Predictor:
    def set_labels(self) -> bool:
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class DetectionNode:
    """Base class for detection node

    Raises
    ------
    NotImplementedError
        Must be implemented by child class
    """

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

    def __init__(self) -> None:
        # placeholder members
        # should be implemented by child class
        self.img_queue = queue.Queue(maxsize=2)
        self.drop_old_msg = True
        self.viz_pub = rospy.Publisher()
        self.detection_pub = rospy.Publisher()
        self.max_marker_count = 1000
        self.target_frame = "target_frame"
        self.source_frame = "source_frame"
        self.debug = True
        self.predictor = Predictor()

        raise NotImplementedError()

    def init_members(self) -> str:
        """Setup node members.

        Returns
        -------
        str
            weight path from rosparam
        """
        # #
        # pub / sub
        # #
        color_sub_topic = rospy.get_param("~input", "/camera/color/image_raw")
        depth_info_sub_topic = rospy.get_param(
            "~depth_aligned_info", "/camera/aligned_depth_to_color/camera_info"
        )
        depth_sub_topic = rospy.get_param(
            "~depth_aligned_img", "/camera/aligned_depth_to_color/image_raw"
        )
        pub_topic = rospy.get_param("~output", "~pred")
        weights = rospy.get_param("~weights")

        self.drop_old_msg = rospy.get_param("~drop", True)
        self.debug = rospy.get_param("~debug", True)
        self.target_frame = rospy.get_param("~target_frame", "map")
        self.source_frame = rospy.get_param(
            "~source_frame", "camera_color_optical_frame"
        )

        # #
        # detection params
        # #
        self.labels = rospy.get_param("~labels", "")
        self.base_labels = rospy.get_param("~base_labels", "")
        self.confidence_thresh = rospy.get_param("~confidence", 0.5)
        self.depth_threshold = rospy.get_param("~depth_threshold", 7.5)
        self.depth_scale = rospy.get_param("~depth_scale", 1000)
        self.detect_period = rospy.get_param("~detect_period", 1e-3)
        self.publish_deprojection = rospy.get_param("~publish_deprojection", True)

        # #
        # setup class members
        # #
        self.labels = self.parse_labels(self.labels)
        self.base_labels = self.parse_labels(self.base_labels)

        rospy.loginfo(f"\n\n\nusing labels: {self.labels} and {self.base_labels}")

        self.bridge = cv_bridge.CvBridge()
        self.img_queue = queue.Queue(maxsize=2)

        self.intrinsics = None
        self.last_depth = None

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # #
        # setup subscribers
        # #
        self.img_sub = rospy.Subscriber(
            color_sub_topic, Image, self.img_callback, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(depth_sub_topic, Image, self.depth_callback)
        self.depth_info_sub = rospy.Subscriber(
            depth_info_sub_topic,
            CameraInfo,
            self.depth_info_callback,
        )

        # #
        # setup publishers
        # #
        self.img_pub = rospy.Publisher(pub_topic, Image, queue_size=1)
        self.detection_pub = rospy.Publisher("~detections", Detection, queue_size=1)

        # #
        # for visualization
        # #
        self.viz_pub = rospy.Publisher("~detections_viz", Marker, queue_size=10)
        self.marker_count = 1000  # deconflict from tracks
        self.max_marker_count = 2000

        self.class_srv = rospy.Service("~set_labels", SetLabels, self.set_labels_cbk)
        self.get_class_srv = rospy.Service("~get_labels", GetLabels, self.get_labels)

        self.setting_labels = False

        return weights

    def parse_labels(self, labels: str) -> List[str]:
        if labels != "":
            labels = labels.split(",")
            labels = [l.strip() for l in labels]
        else:
            labels = []

        return labels

    def set_labels_cbk(self, req: SetLabelsRequest) -> SetLabelsResponse:
        try:
            self.labels = [l.strip() for l in req.labels.split(",")]
            self.set_predictor_labels()
            rospy.loginfo(f"setting labels to: {self.labels}")
            return SetLabelsResponse(success=True)
        except Exception as ex:
            print(ex)
            return SetLabelsResponse(success=False)

    def set_predictor_labels(self):
        prediction_labels = list(set(self.labels + self.base_labels))
        self.setting_labels = True
        self.predictor.set_labels(prediction_labels)
        self.setting_labels = False

    def get_labels(self, req: GetLabelsRequest) -> GetLabelsResponse:
        return GetLabelsResponse(labels=str(self.labels + self.base_labels))

    def spin_node(self):
        """Read from image queue and run detection.

        ROS will drop incoming messages if the subscriber queue is full.
        We want to drop old messages in favor of incoming ones, hence the
        queue logic.
        """
        while True:
            if not self.img_queue.empty():
                if self.img_queue.qsize():
                    img = self.img_queue.get(block=True)

                    if len(self.labels + self.base_labels):
                        self.detect(img)
            rospy.sleep(self.detect_period)

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

    def publish_detection_marker(self, header: Header, position: np.ndarray) -> None:
        marker_msg = create_marker_msg(
            id=self.marker_count,
            header=header,
            position=(position[0], position[1], position[2]),
            scale=0.25,
            color=ColorRGBA(r=0.75, g=0.75, b=0.75, a=1),
        )
        self.viz_pub.publish(marker_msg)

        self.marker_count += 1
        if self.marker_count > self.max_marker_count:
            self.marker_count = 1000

    def publish_detection_msg(
        self,
        class_id: int,
        confidence: float,
        pose: Tuple[float, float, float],
        header: Header,
        label: str,
    ) -> None:
        object_msgs = []
        object_msg = ObjectHypothesisWithPose()
        object_msg.id = 0  # TODO deprecating class ids
        object_msg.score = confidence
        object_msg.pose.pose.position = Point(x=pose[0], y=pose[1], z=pose[2])
        object_msg.pose.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        object_msgs.append(object_msg)
        detection_msg = Detection()
        detection_msg.header = header
        detection_msg.header.frame_id = self.target_frame  # TODO yes?
        detection_msg.results = object_msgs
        detection_msg.labels.append(label)

        self.detection_pub.publish(detection_msg)

    def deproject_detections(
        self, x: float, y: float, w: float, h: float, time: float
    ) -> Tuple[Tuple[float, float, float], float]:
        """Get 3D location of a 2D detection from depth

        Parameters
        ----------
        x : float
            X coordinate (center, image space)
        y : float
            Y coordinate (center, image space)
        w : float
            Detection width
        h : float
            Detection height
        time : float
            Time of detection

        Returns
        -------
        Tuple[Tuple[float, float, float], Float]
            - (x, y, z) location in world coordinates
            - value of corresponding depth image
        """
        if self.last_depth == None or self.intrinsics == None:
            return (0, 0, 0), 0

        # depth is given in mm. We convert that to meters
        depth_img = decode_img_msg(self.last_depth)
        depth_img = depth_img / self.depth_scale

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
            self.target_frame, self.source_frame, rospy.Time()
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

        return (x, y, z), depth_point

    def detect(self, img_msg: Image) -> None:
        """Run inference and publish
        - detections
        - visualizations (if requested)

        Parameters
        ----------
        img_msg : Image
            Incoming image message.
        """
        if self.setting_labels:
            return

        pred_labels = self.labels.copy()

        img = decode_img_msg(img_msg)
        pred_color, classes, boxes, confidences = self.predictor.predict(
            img, plot_output=self.debug
        )

        if self.debug:
            # pred_color = pred[0].plot()
            color_msg = self.bridge.cv2_to_imgmsg(pred_color, encoding="passthrough")
            color_msg.header = img_msg.header  # TODO do we want this?
            color_msg.encoding = "rgb8"

            self.img_pub.publish(color_msg)

        if not self.publish_deprojection:
            for box, class_id, conf in zip(boxes, classes, confidences):
                rospy.logdebug(f"{class_id}: {conf:0.2f}")

            return

        for box, class_id, conf in zip(boxes, classes, confidences):
            if conf < self.confidence_thresh:
                continue

            (x, y, z), depth_point = self.deproject_detections(
                box[::2].mean(),
                box[1::2].mean(),
                w=box[2] - box[0],
                h=box[3] - box[1],
                time=img_msg.header.stamp,
            )

            # dont publish if 0
            if depth_point == 0:
                continue

            # don't publish detections far from camera
            if depth_point > self.depth_threshold:
                continue

            self.publish_detection_msg(
                class_id=class_id,
                confidence=conf,
                pose=(x, y, z),
                header=img_msg.header,
                label=class_id,
            )

            self.publish_detection_marker(header=img_msg.header, position=(x, y, z))
