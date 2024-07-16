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

[//]: # (### Downloading weights)

[//]: # (Download pretrained diffusion models and value functions with:)

[//]: # (```)

[//]: # (./scripts/download_pretrained.sh)

[//]: # (```)

[//]: # ()
[//]: # (This command downloads and extracts a [tarfile]&#40;https://drive.google.com/file/d/1srTq0OFQtWIv9A7fwm3fwh1StA__qr6y/view?usp=sharing&#41; containing [this directory]&#40;https://drive.google.com/drive/folders/1ie6z3toz9OjcarJuwjQwXXzDwh1XnS02?usp=sharing&#41; to `logs/pretrained`. The models are organized according to the following structure:)

[//]: # (```)

[//]: # (└── logs/pretrained)

[//]: # (    ├── ${environment_1})

[//]: # (    │   ├── diffusion)

[//]: # (    │   │   └── ${experiment_name})

[//]: # (    │   │       ├── state_${epoch}.pt)

[//]: # (    │   │       ├── sample-${epoch}-*.png)

[//]: # (    │   │       └── {dataset, diffusion, model, render, trainer}_config.pkl)

[//]: # (    │   └── values)

[//]: # (    │       └── ${experiment_name})

[//]: # (    │           ├── state_${epoch}.pt)

[//]: # (    │           └── {dataset, diffusion, model, render, trainer}_config.pkl)

[//]: # (    ├── ${environment_2})

[//]: # (    │   └── ...)

[//]: # (```)

[//]: # (The `state_${epoch}.pt` files contain the network weights and the `config.pkl` files contain the instantation arguments for the relevant classes.)

[//]: # (The png files contain samples from different points during training of the diffusion model.)

[//]: # (### Planning)

[//]: # ()
[//]: # (To plan with guided sampling, run:)

[//]: # (```)

[//]: # (python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs/pretrained)

[//]: # (```)

[//]: # ()
[//]: # (The `--logbase` flag points the [experiment loaders]&#40;scripts/plan_guided.py#L22-L30&#41; to the folder containing the pretrained models.)

[//]: # (You can override planning hyperparameters with flags, such as `--batch_size 8`, but the default)

[//]: # (hyperparameters are a good starting point.)



[//]: # (2. Train a value function with:)

[//]: # (```)

[//]: # (python scripts/train_values.py --dataset halfcheetah-medium-expert-v2)

[//]: # (```)

[//]: # (See [locomotion:values]&#40;config/locomotion.py#L67-L108&#41; for the corresponding default hyperparameters.)


**Deferred f-strings.** Note that some planning script arguments, such as `--n_diffusion_steps` or `--discount`,
do not actually change any logic during planning, but simply load a different model using a deferred f-string.
For example, the following flags:
```
---horizon 32 --n_diffusion_steps 20 --discount 0.997
--value_loadpath 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}'
```
will resolve to a value checkpoint path of `values/defaults_H32_T20_d0.997`. It is possible to
change the horizon of the diffusion model after training (see [here](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing) for an example),
but not for the value function.

## Docker

1. Build the image:
```
docker build -f Dockerfile . -t diffuser
```

2. Test the image:
```
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
    bash -c \
    "export PYTHONPATH=$PYTHONPATH:/home/code && \
    python /home/code/scripts/train.py --dataset halfcheetah-medium-expert-v2 --logbase logs"
```

[//]: # (## Reference)

[//]: # (```)

[//]: # (@inproceedings{janner2022diffuser,)

[//]: # (  title = {Planning with Diffusion for Flexible Behavior Synthesis},)

[//]: # (  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},)

[//]: # (  booktitle = {International Conference on Machine Learning},)

[//]: # (  year = {2022},)

[//]: # (})

[//]: # (```)


## References

The diffusion model and main branch of this repository is based on Janner et al. [Planning with Diffusion for Flexible Behavior Synthesis](https://github.com/jannerm/diffuser).
The Cyclical GAN is based on [Tensorflow CycleGAN tutorials](https://www.tensorflow.org/tutorials/generative/cyclegan).
We use [YOLOv5 from ultralytics](https://github.com/ultralytics/yolov5) for object detection.
