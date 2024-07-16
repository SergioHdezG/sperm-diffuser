# Real-like Synthetic video of sperm with generative models &nbsp;&nbsp; 

This repository is organized in three brands.

- The [main branch](https://github.com/SergioHdezG/sperm-diffuser) contains the diffusion model to generate schematic sperm videos. It includes the pipeline to generate individual spermatozoon trajectories and annotated videos of multiple schematic spermatozoa. This branch makes use of a modified version of the diffusion model proposed by Janner et al. [[Planning with Diffusion for Flexible Behavior Synthesis](https://github.com/jannerm/diffuser)].
- The [style-transfer branch](https://github.com/SergioHdezG/tree/style-transfer) contains tools to transform the schematic videos generated with the diffusion model into real-like style. This branch makes use of a Cyclical Generative Adversarial Network to perform the style transfer.
- The [sperm-detection branch](https://github.com/SergioHdezG/tree/sper-detection) consist of a fork of [YOLOv5 from ultralytics](https://github.com/ultralytics/yolov5) from ultralitycs adapted to perform our evaluation pipeline on sperm detection.
<p align="center">
    <img src="https://github.com/SergioHdezG/sperm-diffuser/blob/main/images/spermdiffuserabstract.png" width="60%" title="Abstract">
</p>


[//]: # (## Installation)

[//]: # ()
[//]: # (```)

[//]: # (conda env create -f environment.yml)

[//]: # (conda activate diffuser)

[//]: # (pip install -e .)

[//]: # (```)

## Training the model

We provide the real data parameterized using our sperm model in [diffser/datasets/BezierSplinesData](https://github.com/SergioHdezG/sperm-diffuser/diffuser/datasets/BezierSplinesData).
This dataset is split in progressive, slow progressive and inmotile sperm. The next program enables the training of a model:

```
python scripts/train_sperm -dataset SingleSpermBezierIncrementsDataAugSimplified-v0 --logbase logs ...
```

The default hyperparameters are listed in [locomotion:diffusion](config/locomotion.py#L22-L65).
You can override any of them with flags, eg, `--n_diffusion_steps 100`.

## Generate individual spermatozoa trajectories

Two ways to generate new sperm trajectories are provided:

1. If given an initial condition coming from the real dataset ([BezierSplinesData](https://github.com/SergioHdezG/sperm-diffuser/diffuser/datasets/BezierSplinesData)), the model generate new trajectories from the given real initial condition:

```
python scripts/paper_images/generate_sperms_jsons_rans_larger_real_init_condition.py -dataset SingleSpermBezierIncrementsDataAugSimplified-v0 -dataset diffuser/datasets/BezierSplinesData/slow
```

2. If no access to the real conditions is available, we allow generation given a random sampled initial condition. These conditions are sampled from a multivariate gaussian distribution fitted to the values of the parameters in the real dataset ([BezierSplinesData](https://github.com/SergioHdezG/sperm-diffuser/diffuser/datasets/BezierSplinesData)). The next program generates a complete dataset of synthetic spermatozoa using the trained diffusion models for progressive, slow progressive and inmotile sperm.

```
python scripts/paper_images/generate_sperms_jsons_dataset_gauss_init.py -dataset SingleSpermBezierIncrementsDataAugSimplified-v0 -dataset ...
```

## Reproducibility

Next scripts enable to replicate the paper experiments.

1. Training models on "progresive", "slow progresive" and "inmotile" datasets.
```
python scripts/train_sperm.py --horizon 16 --sample_freq 250 --diffusion models.GaussianDiffusionImitationCondition --n_train_steps 10000 --n_steps_per_epoch 2000 --save_freq 1000 --action_weight 0 --loader datasets.SequenceDatasetSpermNormalized --loss_type sperm_loss --renderer utils.EMARenderer --n_diffusion_steps 20 --learning_rate 2e-5 ----data_file diffuser/datasets/BezierSplinesData/progressive
python scripts/train_sperm.py --horizon 16 --sample_freq 250 --diffusion models.GaussianDiffusionImitationCondition --n_train_steps 10000 --n_steps_per_epoch 2000 --save_freq 1000 --action_weight 0 --loader datasets.SequenceDatasetSpermNormalized --loss_type sperm_loss --renderer utils.EMARenderer --n_diffusion_steps 20 --learning_rate 2e-5 ----data_file diffuser/datasets/BezierSplinesData/slow
python scripts/train_sperm.py --horizon 16 --sample_freq 250 --diffusion models.GaussianDiffusionImitationCondition --n_train_steps 10000 --n_steps_per_epoch 2000 --save_freq 1000 --action_weight 0 --loader datasets.SequenceDatasetSpermNormalized --loss_type sperm_loss --renderer utils.EMARenderer --n_diffusion_steps 20 --learning_rate 2e-5 ----data_file diffuser/datasets/BezierSplinesData/inmotile
```


## Docker

1. Build the image:
```
docker build -f Dockerfile . -t diffuser
```


