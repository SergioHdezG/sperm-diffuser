# Real-like synthetic sperm video generation from learnt behaviours &nbsp;&nbsp; 

This branch is a fork of [yolov5](https://github.com/ultralytics/yolov5) from Ultralytics.

This repository is organized in three branches.

- The [main branch](https://github.com/SergioHdezG/sperm-diffuser) contains the diffusion model to generate schematic sperm videos. It includes the pipeline to generate individual spermatozoon trajectories and annotated videos of multiple schematic spermatozoa. This branch makes use of a modified version of the diffusion model proposed by Janner et al. [[Planning with Diffusion for Flexible Behavior Synthesis](https://github.com/jannerm/diffuser)].
- The [style-transfer branch](https://github.com/SergioHdezG/sperm-diffuser/tree/style-transfer) contains tools to transform the schematic videos generated with the diffusion model into real-like style. This branch makes use of a Cyclical Generative Adversarial Network to perform the style transfer.
- The [sperm-detection branch](https://github.com/SergioHdezG/sperm-diffuser/tree/sperm-detection) consist of a fork of [YOLOv5 from ultralytics](https://github.com/ultralytics/yolov5) from ultralitycs adapted to perform our evaluation pipeline on sperm detection.
<p align="center">
    <img src="https://github.com/SergioHdezG/sperm-diffuser/blob/main/images/abstract_spermdiffuser_v2.png" width="99%" title="Abstract">
    <br>
    <em>Figure:</em> Upper row summarizes the process to obtain individual parametrized trajectories of spermatozoa by applying classic computer vision techniques. Bottom row shows the proposed method to generate individual spermatozoa trajectories from a noisy input with a diffusion model and the subsequent style transfer procedure.
</p>

# Installation

```bash
pip install ultralytics
```

or clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.8.0**](https://www.python.org/) environment, including
[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

# Reproducibility

Train models:
```
sh run_train1.sh
sh run_train2.sh
sh run_train2.sh
```
Test the models:
```
sh run_val1.sh
sh run_val2.sh
```
