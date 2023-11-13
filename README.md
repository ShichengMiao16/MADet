# [MADet: Mutual-Assistance Learning for Object Detection](https://ieeexplore.ieee.org/abstract/document/10265160)

## Introduction

![demo image](resources/madet.jpg)

Object detection is a fundamental yet challenging task in computer vision. Despite the great strides made over recent years, modern detectors may still produce unsatisfactory performance due to certain factors, such as non-universal object features and single regression manner. In this paper, we draw on the idea of mutual-assistance (MA) learning and accordingly propose a robust one-stage detector, referred as MADet, to address these weaknesses. First, the spirit of MA is manifested in the head design of the detector. Decoupled classification and regression features are reintegrated to provide shared offsets, avoiding inconsistency between feature-prediction pairs induced by zero or erroneous offsets. Second, the spirit of MA is captured in the optimization paradigm of the detector. Both anchor-based and anchor-free regression fashions are utilized jointly to boost the capability to retrieve objects with various characteristics, especially for large aspect ratios, occlusion from similar-sized objects, etc. Furthermore, we meticulously devise a quality assessment mechanism to facilitate adaptive sample selection and loss term reweighting. Extensive experiments on standard benchmarks verify the effectiveness of our approach. On MS-COCO, MADet achieves 42.5% AP with vanilla ResNet50 backbone, dramatically surpassing multiple strong baselines and setting a new state of the art.

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MADet.

## Results and models

### MS-COCO dataset

| Backbone | Lr schd | box AP |                           Baidu Yun                          |                                           Google Drive                                         |
|:--------:|:-------:|:------:|:------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
|  R50-FPN |    1x   |  42.5  | [key: 71gj](https://pan.baidu.com/s/18mULqHGtnIAZv8n5G1xYhg) | [model](https://drive.google.com/file/d/17g3VxwXTs7Mf2XdMt2WQ4NIlP_tdpgI_/view?usp=drive_link) |

## Acknowledgement

The implementation of MADet is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation
```
@ARTICLE{MADet,
  title = {Mutual-Assistance Learning for Object Detection},
  author = {Xie, Xingxing and Lang, Chunbo and Miao, Shicheng and Cheng, Gong and Li, Ke and Han, Junwei},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2023},
  pages = {1-14},
  doi = {10.1109/TPAMI.2023.3319634}
}
```
