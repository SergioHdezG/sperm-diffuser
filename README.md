# Real-like Synthetic video of sperm with generative models &nbsp;&nbsp; 

This repository is organized in three brands.

- The [main branch](https://github.com/SergioHdezG/sperm-diffuser) contains the diffusion model to generate schematic sperm videos. It includes the pipeline to generate individual spermatozoon trajectories and annotated videos of multiple schematic spermatozoa. This branch makes use of a modified version of the diffusion model proposed by Janner et al. [[Planning with Diffusion for Flexible Behavior Synthesis](https://github.com/jannerm/diffuser)].
- The [style-transfer branch](https://github.com/SergioHdezG/tree/style-transfer) contains tools to transform the schematic videos generated with the diffusion model into real-like style. This branch makes use of a Cyclical Generative Adversarial Network to perform the style transfer.

<p align="center">
    <img src="https://github.com/SergioHdezG/sperm-diffuser/blob/main/images/spermdiffuserabstract.pdf" width="60%" title="Abstract">
</p>


[//]: # (## Installation)

[//]: # ()
[//]: # (```)

[//]: # (conda env create -f environment.yml)

[//]: # (conda activate diffuser)

[//]: # (pip install -e .)

[//]: # (```)

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

## Training the model

We provide the real data parameterized using our sperm model in [diffser/datasets/BezierSplinesData](https://github.com/SergioHdezG/sperm-diffuser/diffuser/datasets/BezierSplinesData).
This dataset is split in progressive, slow progressive and inmotile sperm. The next program enables the training of a model:

```
python scripts/train_sperm -dataset SingleSpermBezierIncrementsDataAugSimplified-v0 --logbase logs ...
```
The default hyperparameters are listed in [locomotion:diffusion](config/locomotion.py#L22-L65).
You can override any of them with flags, eg, `--n_diffusion_steps 100`.

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

## Singularity

1. Build the image:
```
singularity build --fakeroot diffuser.sif Singularity.def
```

2. Test the image:
```
singularity exec --nv --writable-tmpfs diffuser.sif \
        bash -c \
        "pip install -e . && \
        python scripts/train.py --dataset halfcheetah-medium-expert-v2 --logbase logs"
```


## Running on Azure

#### Setup

1. Tag the Docker image (built in the [Docker section](#Docker)) and push it to Docker Hub:
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag diffuser ${DOCKER_USERNAME}/diffuser:latest
docker image push ${DOCKER_USERNAME}/diffuser
```

3. Update [`azure/config.py`](azure/config.py), either by modifying the file directly or setting the relevant [environment variables](azure/config.py#L47-L52). To set the `AZURE_STORAGE_CONNECTION` variable, navigate to the `Access keys` section of your storage account. Click `Show keys` and copy the `Connection string`.

4. Download [`azcopy`](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10): `./azure/download.sh`

#### Usage

Launch training jobs with `python azure/launch.py`. The launch script takes no command-line arguments; instead, it launches a job for every combination of hyperparameters in [`params_to_sweep`](azure/launch_train.py#L36-L38).


#### Viewing results

To rsync the results from the Azure storage container, run `./azure/sync.sh`.

To mount the storage container:
1. Create a blobfuse config with `./azure/make_fuse_config.sh`
2. Run `./azure/mount.sh` to mount the storage container to `~/azure_mount`

To unmount the container, run `sudo umount -f ~/azure_mount; rm -r ~/azure_mount`


## Reference
```
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```


## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.
