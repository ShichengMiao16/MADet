# Get Started

This page provides basic tutorials about the usage of MADet.
For installation instructions, please see [install.md](install.md).

## Prepare datasets

It is recommended to download and extract the dataset somewhere outside the project directory and symlink the dataset root to `$MADet/data` as below.
If your folder structure is different, you may need to change the corresponding paths in config files.
Recommended folder structure is given as follows.

```plain
MADet
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

## Training and testing of models for object detection

### Training

You can use the following commands to train a model.

```shell
# single-gpu training
cd MADet
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multi-gpu training
cd MADet
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### Testing

You can use the following command to test a model.

```shell
cd MADet
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

## Demos

We provide demo scripts to test images.

### Image demo

```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}]
```

### Webcam demo

```shell
python demo/webcam_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--camera-id ${CAMERA-ID}] \
    [--score-thr ${SCORE_THR}]
```

## Useful tools

Some useful tools are available [here](useful_tools.md).