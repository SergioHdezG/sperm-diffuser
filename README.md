# Real-like synthetic sperm video generation from learned behaviors  &nbsp;&nbsp; 

<p align="center">
    <img src="https://github.com/SergioHdezG/sperm-diffuser/blob/main/images/synthTest1.gif" width="40%" title="GIF 1">
    <img src="https://github.com/SergioHdezG/sperm-diffuser/blob/main/images/synthTest2.gif" width="40%" title="GIF 2">
</p>

<p align="center"><em>Figure: Sequence of videos showing schematic frames modelled using the diffusion model output, alongside their photorealistic counterpart after style transfer.</em></p>


This branch is a fork of the tensorflow 2 implementation of CycleGAN: [CycleGAN-Tensorflow-2](https://github.com/LynnHo/CycleGAN-Tensorflow-2). For deeper information on how to use it please go to the original repository.

This repository is organized in three branches.

- The [main branch](https://github.com/SergioHdezG/sperm-diffuser) contains the diffusion model to generate schematic sperm videos. It includes the pipeline to generate individual spermatozoon trajectories and annotated videos of multiple schematic spermatozoa. This branch makes use of a modified version of the diffusion model proposed by Janner et al. [[Planning with Diffusion for Flexible Behavior Synthesis](https://github.com/jannerm/diffuser)].
- The [style-transfer branch](https://github.com/SergioHdezG/sperm-diffuser/tree/style-transfer) contains tools to transform the schematic videos generated with the diffusion model into real-like style. This branch makes use of a Cyclical Generative Adversarial Network to perform the style transfer.
- The [sperm-detection branch](https://github.com/SergioHdezG/sperm-diffuser/tree/sperm-detection) consist of a fork of [YOLOv5 from ultralytics](https://github.com/ultralytics/yolov5) from ultralitycs adapted to perform our evaluation pipeline on sperm detection.

The synthetically generated dataset is available in [dataset/synthetic_sperm_dataset.zip](https://github.com/SergioHdezG/sperm-diffuser/blob/main/synthetic_sperm_dataset.zip)

<p align="center">
    <img src="https://github.com/SergioHdezG/sperm-diffuser/blob/main/images/abstract_spermdiffuser_v2.png" width="99%" title="Abstract">
    <br>
    <em>Figure:</em> Upper row summarizes the process to obtain individual parametrized trajectories of spermatozoa by applying classic computer vision techniques. Bottom row shows the proposed method to generate individual spermatozoa trajectories from a noisy input with a diffusion model and the subsequent style transfer procedure.
</p>


# Installation

```console
conda create -n tensorflow-2.7 python=3.8
source activate tensorflow-2.7
conda install scikit-image tqdm tensorflow-gpu=2.7
pip install tensorflow-addons==0.15.0
```

# Use the trained model

```
python test.py --experiment_dir output/november/SynthBezier2RealImg --checkpoint_dir output/november/SynthBezier2RealImg3/checkpoints_BCK2
```

# Train a new model

Note that we are not allowed to share the original real images, therefore, we include a folder with real-looking synthetic images only for demonstration purposes. 
For further information on how to use other features of the CycleGAN implementation we refer to [CycleGAN-Tensorflow-2](https://github.com/LynnHo/CycleGAN-Tensorflow-2).
```
python train.py --output_dir november/SynthBezier2RealImg3 --dataset SynthBezier2RealImg --trainA trainFullBezierImage --trainB trainFullSpermImage --testA testFullBezierImage --testB testFullSpermImage
```
