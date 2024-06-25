# lang-seg-ros

ROS wrapper for running open vocabulary detection and segmentation models.

## Supported models
- [GroundingDino](https://arxiv.org/abs/2303.05499)
- [Open Vocab YOLO](https://github.com/ultralytics/ultralytics)
- [language-driven semantic segmentation](https://arxiv.org/abs/2201.03546)

## Requirements

(assumes you have a basic ROS install)

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
