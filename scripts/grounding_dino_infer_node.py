#!/usr/bin/env python
import os
import queue
import time
from typing import Tuple, Union
from pathlib import Path

import cv2
import cv_bridge
import numpy as np
import pyrealsense2 as rs2
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Quaternion
from lang_seg_ros.viz_utils import create_marker_msg
from rosbags.typesys.types import sensor_msgs__msg__CompressedImage as CompressedImgMsg
from rosbags.typesys.types import sensor_msgs__msg__Image as ImageMsg
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker

try:
    import dataclasses

    import numpy as np
    import supervision as sv
    import torch
    import torchvision
    from groundingdino.util.inference import Model as GDModel
    from supervision.draw.color import Color, ColorPalette


except ImportError as ex:
    raise ValueError(f"must install grouning dino: {ex}")

GSA_PATH = "/home/zac/projects/foundation-models/Grounded-Segment-Anything/"
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")


# should go in utils
def vis_result_fast(
    image: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    color: Union[Color, ColorPalette] = ColorPalette.default(),
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    """
    Annotate the image with the detection results.
    This is fast but of the same resolution of the input image, thus can be blurry.
    """
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        color=color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(color=color)

    if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None:
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(confidences, class_ids)
            ]
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print(
            "Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing."
        )

    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    if draw_bbox:
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
    return annotated_image, labels


class Dino:
    def __init__(
        self,
        classes,
        confidence,
        ckpt_path=GROUNDING_DINO_CHECKPOINT_PATH,
        config_path=GROUNDING_DINO_CONFIG_PATH,
    ):
        self.grounding_dino_model = GDModel(
            model_config_path=str(config_path),
            model_checkpoint_path=str(ckpt_path),
            device="cuda",
        )
        self.classes = classes
        self.confidence = confidence
        self.grounding_dino_model.model.eval()

    def predict(self, img, debug_msg=True):
        t1 = time.time()
        detections = self.grounding_dino_model.predict_with_classes(
            img,
            self.classes,
            box_threshold=self.confidence,
            text_threshold=self.confidence,
        )
        rospy.loginfo(f"GD prediction took: {(time.time() - t1):0.2}s")

        annotated_image = img

        if len(detections.class_id) > 0:
            ### Non-maximum suppression ###
            # print(f"Before NMS: {len(detections.xyxy)} boxes")
            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    0.5,
                )
                .numpy()
                .tolist()
            )

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            # Somehow some detections will have class_id=-1, remove them
            valid_idx = detections.class_id != -1
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]

            if debug_msg:
                annotated_image, labels = vis_result_fast(
                    img, detections, self.classes, instance_random_color=True
                )

            return (
                annotated_image,
                detections.class_id,
                detections.xyxy,
                detections.confidence,
            )
        else:
            return img, np.array([]), np.array([]), np.array([])


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
        self.source_frame = rospy.get_param(
            "~source_frame", "camera_color_optical_frame"
        )

        self.labels = rospy.get_param("~labels", "")
        self.confidence_thresh = rospy.get_param("~confidence", 0.5)
        self.depth_threshold = rospy.get_param("~depth_threshold", 7.5)
        self.depth_scale = rospy.get_param("~depth_scale", 1000)
        self.detect_period = rospy.get_param("~detect_period", 1e-3)
        self.publish_deprojection = rospy.get_param("~publish_deprojection", True)

        # setup class members
        if self.labels != "":
            self.labels = self.labels.split(",")
            self.labels = [l.strip() for l in self.labels]
        else:
            self.labels = self.DEFAULT_LABELS
        rospy.loginfo(f"using labels: {self.labels}")

        self.bridge = cv_bridge.CvBridge()
        self.img_queue = queue.Queue(maxsize=2)

        self.dino_infer = Dino(
            classes=self.labels,
            confidence=self.confidence_thresh,
            ckpt_path=Path(weights) / "groundingdino_swint_ogc.pth",
            config_path=Path(weights) / "GroundingDINO_SwinT_OGC.py",
        )
        rospy.loginfo(" loaded.")

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

        # for visualization
        self.viz_pub = rospy.Publisher("~detections_viz", Marker, queue_size=10)
        self.marker_count = 1000  # deconflict from tracks
        self.max_marker_count = 2000

        # ROS will drop incoming messages if subscriber is full. It would
        # be preferable to drop old messages, so we'll use a queue as
        # an intermediate container

        # TODO still an issue where queue is read too fast
        while True:
            if not self.img_queue.empty():
                if self.img_queue.qsize():
                    img = self.img_queue.get(block=True)
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

    def detect(self, img_msg: Image) -> None:
        img = decode_img_msg(img_msg)
        pred_color, classes, boxes, confidences = self.dino_infer.predict(
            img, debug_msg=self.debug
        )

        if self.debug:
            # pred_color = pred[0].plot()
            color_msg = self.bridge.cv2_to_imgmsg(pred_color, encoding="passthrough")
            color_msg.header = img_msg.header  # TODO do we want this?
            color_msg.encoding = "rgb8"

            self.img_pub.publish(color_msg)

        # boxes = pred[0].boxes.xyxy.cpu().numpy()
        # classes = pred[0].boxes.cls.cpu().numpy().astype(np.int16)
        # confidences = pred[0].boxes.conf.cpu().numpy()

        if not self.publish_deprojection:
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

            # don't publish detections far from camera
            if depth_point > self.depth_threshold:
                continue

            self.publish_detection_msg(
                class_id=class_id,
                confidence=conf,
                pose=(x, y, z),
                header=img_msg.header,
            )

            self.publish_detection_marker(header=img_msg.header, position=(x, y, z))

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


if __name__ == "__main__":
    rospy.init_node("lang_seg_ros")

    infer = LangSegInferRos()
