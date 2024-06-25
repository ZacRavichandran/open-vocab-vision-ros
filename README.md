# lang-seg-ros

ROS wrapper for running open vocabulary detection and segmentation models.

## Supported models
- [GroundingDino](https://arxiv.org/abs/2303.05499)
- [Open Vocab YOLO](https://github.com/ultralytics/ultralytics)
- [language-driven semantic segmentation](https://arxiv.org/abs/2201.03546)

## Installation

(assumes you have a basic ROS install).

Each supported model can be installed independently (ie if you don't want to use one, you don't have to isnstall it). 

### Grounding dino

**Note**: This will try to install torch. It may be preferable to install torch manually, so you can ensure the right cuda, etc. 

Install Grounding Dino (forked from https://github.com/IDEA-Research/Grounded-Segment-Anything)
```sh
git clone https://github.com/ZacRavichandran/Grounded-Segment-Anything
cd Grounded-Segment-Anything/GroundingDino
python -m pip install -f requirements.txt
python -m pip install -e .
```

## YOLO
Just install ultralytics
``sh
python -m pip install ultralytics
``

### Lang seg
* install [lang-seg](https://github.com/ZacRavichandran/lang-seg)
* download [checkpoint](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing)

## Running

Please see the [launch files](./launch) for example configurations.


```
Node [/grounding_dino_ros]
 * /grounding_dino_ros/confidence: Confidence threshold
 * /grounding_dino_ros/depth_scale: Divide depth image by this to get metric value (camera dependant) 
 * /grounding_dino_ros/depth_threshold: Ignore detections beyond this depth.
 * /grounding_dino_ros/detect_period: Wait at least this long between inference.
 * /grounding_dino_ros/drop: Drop old messages, if needed.
 * /grounding_dino_ros/input: Input image topic
 * /grounding_dino_ros/labels: List of labels
 * /grounding_dino_ros/output: Output topic
 * /grounding_dino_ros/publish_deprojection: Publish 3D deprojection
 * /grounding_dino_ros/weights: Path to weights
```






