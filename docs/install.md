## Prerequisites

- Linux
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

The compatible MADet and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

|    MADet version    |     MMCV version    |
|:-------------------:|:-------------------:|
| master              | mmcv-full>=1.2.4, <1.3|
| 2.8.0               | mmcv-full>=1.2.4, <1.3|
| 2.7.0               | mmcv-full>=1.1.5, <1.3|
| 2.6.0               | mmcv-full>=1.1.5, <1.3|
| 2.5.0               | mmcv-full>=1.1.5, <1.3|
| 2.4.0               | mmcv-full>=1.1.1, <1.3|
| 2.3.0               | mmcv-full==1.0.5    |
| 2.3.0rc0            | mmcv-full>=1.0.2    |
| 2.2.1               | mmcv==0.6.2         |
| 2.2.0               | mmcv==0.6.2         |
| 2.1.0               | mmcv>=0.5.9, <=0.6.1|
| 2.0.0               | mmcv>=0.5.1, <=0.5.8|

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.
We use MADet version 2.8.0 and MMCV version 1.2.7 to successfully install the packages.

## Installation

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n env_name python=3.7 -y
    conda activate env_name
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

    ```shell
    conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt pacakge,
    you can use more CUDA versions such as 9.0.
    
    Note: We use CUDA 9.2, pytorch 1.3.1 and torchvision 0.4.2 to successfully install the virtual environment.
    This set of versions can be used as a reference when you install Pytorch and torchvision.

3. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```
   
   Note: We successfully install mmcv-full 1.2.7 with pytorch 1.3.1, torchvison 0.4.2 and CUDA 9.2.
   This set of versions can be a reference when you install mmcv-full.

4. Clone the MADet repository.

    ```shell
    git clone https://github.com/ShichengMiao16/MADet.git --recursive
    cd MADet
    ```

5. Install build requirements and then install MADet.

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

Note:

a. Following the above instructions, MADet is installed on `dev` mode
, any local modifications made to the code will take effect without the need to reinstall it.

b. If you would like to use `opencv-python-headless` instead of `opencv
-python`, you can install it before installing MMCV.

c. Some dependencies are optional. Simply running `pip install -v -e .` will
 only install the minimum runtime requirements. To use optional dependencies 
 like `albumentations` and `imagecorruptions` either install them manually 
 with `pip install -r requirements/optional.txt` or specify desired extras 
 when calling `pip` (e.g. `pip install -v -e .[optional]`). 
 Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.
 
d. If you install torchvision < 0.5.0, please run `pip install "pillow<9"` to avoid `ImportError`.

### Install with CPU only

The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/webcam_demo.py for example.
However some functionality is gone in this mode:

- Deformable Convolution
- Modulated Deformable Convolution
- RoI pooling
- Deformable RoI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- SyncBatchNorm
- CrissCrossAttention: Criss-Cross Attention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps