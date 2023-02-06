# MADet: Mutual-Assistance Learning for Object Detection

## Introduction

![demo image](resources/madet.jpg)

Object detection aims at locating and identifying potential objects of interest within the given image, 
which can be considered a long-standing fundamental task in the computer vision community. Despite the 
great strides made over recent years, modern detectors may still produce unsatisfactory performance due 
to certain factors, such as inappropriate structural design and single regression manner. In this paper, 
we draw on the idea of mutual-assistance (MA) learning and accordingly propose a robust one-stage detector, 
referred as MADet, to address these weaknesses. First, the spirit of MA is manifested in the head design 
of the detector. Decoupled classification and regression features are reintegrated to provide shared offsets, 
avoiding inconsistency between feature-prediction pairs induced by zero or erroneous offsets. Second, the spirit 
of MA is captured in the optimization paradigm of the detector. Both anchor-based and anchor-free regression 
fashions are utilized jointly to boost the capability to retrieve objects with various characteristics, especially for 
large aspect ratios, occlusion from similar-sized objects, etc. Furthermore, we meticulously devise a quality 
assessment mechanism to facilitate adaptive sample selection and loss term reweighting. Extensive experiments on
standard benchmarks verify the effectiveness of our proposed approach. on MS-COCO, and MADet achieves 42.5% AP 
with vanilla ResNet-50 backbone, dramatically surpassing multiple strong baselines and setting a new state-of-the-art.

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MADet.

## Acknowledgement

The implementation of MADet is based on [MMDetection](https://github.com/open-mmlab/mmdetection).

## License

This project is released under the [Apache 2.0 license](LICENSE).