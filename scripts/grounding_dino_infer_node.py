#!/usr/bin/env python
import os
import queue
from pathlib import Path

import cv_bridge
import rospy
from open_vocab_vision_ros.detection_node import DetectionNode

try:
    from open_vocab_vision_ros.grounding_dino import GroundingDinoInfer
except ImportError as ex:
    raise ValueError(f"must install grouning dino: {ex}")


class DinoInferNode(DetectionNode):
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.img_queue = queue.Queue(maxsize=2)

        weights = self.init_members()

        self.predictor = GroundingDinoInfer(
            classes=self.labels,
            confidence=self.confidence_thresh,
            ckpt_path=Path(weights) / "groundingdino_swint_ogc.pth",
            config_path=Path(weights) / "GroundingDINO_SwinT_OGC.py",
        )
        self.set_predictor_labels()
        rospy.loginfo(" loaded.")
        self.spin_node()


if __name__ == "__main__":
    rospy.init_node("grounding_dino_node")

    infer = DinoInferNode()
