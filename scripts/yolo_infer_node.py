#!/usr/bin/env python
from typing import List, Tuple

import numpy as np
import rospy
from open_vocab_vision_ros.detection_node import DetectionNode

try:
    from ultralytics import YOLO
except ImportError:
    raise ValueError("must install ultralytics")


class YoloInfer:
    def __init__(self, weights: str, labels: List[str]) -> None:
        self.yolo_infer = YOLO(weights)
        self.yolo_infer.set_classes(labels)
        self.labels = labels

    def predict(
        self, img: np.ndarray, plot_output: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred_labels = self.labels.copy()
        self.yolo_infer.set_classes(pred_labels)
        pred = self.yolo_infer.predict(img)

        pred_color = pred[0].plot() if plot_output else img
        boxes = pred[0].boxes.xyxy.cpu().numpy()
        classes = pred[0].boxes.cls.cpu().numpy().astype(np.int16)
        confidences = pred[0].boxes.conf.cpu().numpy()

        cls_val = [pred_labels[c] for c in classes]

        return pred_color, cls_val, boxes, confidences

    def set_labels(self, labels: List[str]) -> bool:
        self.yolo_infer.set_classes(labels)
        self.labels = labels
        return True


class YoloInferNode(DetectionNode):

    def __init__(self):
        weights = self.init_members()
        self.predictor = YoloInfer(weights=weights, labels=self.labels)
        rospy.loginfo("yolo loaded.")

        self.spin_node()


if __name__ == "__main__":
    rospy.init_node("yolo_infer_node")

    infer = YoloInferNode()
