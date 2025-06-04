<h1 align="left" style="display: flex; justify-content: space-between; align-items: center;">
  Real-like synthetic sperm video generation from learned behaviors
  <a href="https://doi.org/10.1007/s10489-025-06407-3" target="_blank">
    <img src="https://img.shields.io/badge/View%20Paper-DOI-blue?style=for-the-badge&logo=springer" alt="View Paper DOI">
  </a>
</h1>

<p align="center">
    <img src="images/synthTest1.gif" width="40%" title="GIF 1">
    <img src="images/synthTest2.gif" width="40%" title="GIF 2">
</p>

<p align="center"><em>Figure: Sequence of videos showing schematic frames modelled using the diffusion model output, alongside their photorealistic counterpart after style transfer.</em></p>

This repository is organized in three branches.

- The [main branch](https://github.com/SergioHdezG/sperm-diffuser) contains the diffusion model to generate schematic sperm videos. It includes the pipeline to generate individual spermatozoon trajectories and annotated videos of multiple schematic spermatozoa. This branch makes use of a modified version of the diffusion model proposed by Janner et al. [[Planning with Diffusion for Flexible Behavior Synthesis](https://github.com/jannerm/diffuser)].
- The [style-transfer branch](https://github.com/SergioHdezG/sperm-diffuser/tree/style-transfer) contains tools to transform the schematic videos generated with the diffusion model into real-like style. This branch makes use of a Cyclical Generative Adversarial Network to perform the style transfer.
- The [sperm-detection branch](https://github.com/SergioHdezG/sperm-diffuser/tree/sperm-detection) consist of a fork of [YOLOv5 from ultralytics](https://github.com/ultralytics/yolov5) from ultralitycs adapted to perform our evaluation pipeline on sperm detection.

The synthetically generated dataset is available in [dataset/synthetic_sperm_dataset.zip](https://github.com/SergioHdezG/sperm-diffuser/tree/main/dataset)

<p align="center">
    <img src="https://github.com/SergioHdezG/sperm-diffuser/blob/main/images/abstract_spermdiffuser_v2.png" width="99%" title="Abstract">
    <br>
    <em>Figure:</em> Upper row summarizes the process to obtain individual parametrized trajectories of spermatozoa by applying classic computer vision techniques. Bottom row shows the proposed method to generate individual spermatozoa trajectories from a noisy input with a diffusion model and the subsequent style transfer procedure.
</p>


## Installation

```
conda env create -f environment.yml
conda activate sperm_diffuser
pip install -e .
```

## Training the model

We provide the real data parameterized using our sperm model in [diffuser/datasets/BezierSplinesData](https://github.com/SergioHdezG/sperm-diffuser/tree/main/diffuser/datasets/BezierSplinesData).
This dataset is split in progressive, slow progressive and inmotile sperm. The next program enables the training of a model:

```
python scripts/train_sperm -dataset SingleSpermBezierIncrementsDataAugSimplified-v0 --logbase logs ...
```

The default hyperparameters are listed in [config/spermFeb.py](https://github.com/SergioHdezG/sperm-diffuser/blob/main/config/spermFeb.py).
You can override any of them with flags, eg, `--n_diffusion_steps 100`.

## Generate dataset 

Next script generate sperm trajectories organized in different folders to create a dataset of schematic videos and its corresponding labels. 

Adjust implicit parameters in the script to obtain the desired results:
- **mean_n_sperms:** mean number of spermatozoa per video
- **std_n_sperm:** standard deviation of number of spermatozoa per video
- **n_sequences:** number of videos to generate
- **seq_len:** seq_len * horizon denotes the length of the videos to be generated
- **make_subfolders (bool):** if True an independent folder is created for each video, if False all the sequences are store inside a folder.

Run the generation process given an initial gaussian condition:
```
python scripts/generate_sperms_jsons_dataset_gauss_init.py --dataset SingleSpermBezierIncrementsDataAugSimplified-v0 --logbase logs --diffusion_loadpath diffusion/defaults_H16_T20/progressive_model --policy sampling.DiffPolicy --horizon 16
```



## Generate Schematic videos and YOLO labels


Adjust implicit parameters in the script to obtain the desired results:
- **resize_img:** tuple (h, w) or None to use original video resolution (1024, 1280)
- **only_one_class (bool):** if True all the bboxes are labeled as class '0', if False progressive, slow and inmotile sperm ara labeled as classes {'0', '1','2'}
- **spline_path:** path to load the parameters of the generated spermatozoa.
- **save_train, save_train_labels_yolo, save_train_labels_complete_csv:** path to save images, yolo labels and csv containing all the parameters of the spermatozoon model.

```
python scripts/generate_yolo_labels.py 
```

## Style transfer and YOLO training

In order to obtain photorealistic frames use the procedure provided in [style-transfer branch](https://github.com/SergioHdezG/sperm-diffuser/tree/style-transfer).
In order to train a YOLO model use [sperm-detection branch](https://github.com/SergioHdezG/sperm-diffuser/tree/sperm-detection).

## Reproducibility

Next scripts enable to replicate the paper experiments.

1. Training models on "progresive", "slow progresive" and "inmotile" datasets.
```
python scripts/train_sperm.py --horizon 16 --sample_freq 250 --diffusion models.GaussianDiffusionImitationCondition --n_train_steps 10000 --n_steps_per_epoch 2000 --save_freq 1000 --action_weight 0 --loader datasets.SequenceDatasetSpermNormalized --loss_type sperm_loss --renderer utils.EMARenderer --n_diffusion_steps 20 --learning_rate 2e-5 ----data_file diffuser/datasets/BezierSplinesData/progressive
python scripts/train_sperm.py --horizon 16 --sample_freq 250 --diffusion models.GaussianDiffusionImitationCondition --n_train_steps 10000 --n_steps_per_epoch 2000 --save_freq 1000 --action_weight 0 --loader datasets.SequenceDatasetSpermNormalized --loss_type sperm_loss --renderer utils.EMARenderer --n_diffusion_steps 20 --learning_rate 2e-5 ----data_file diffuser/datasets/BezierSplinesData/slow
python scripts/train_sperm.py --horizon 16 --sample_freq 250 --diffusion models.GaussianDiffusionImitationCondition --n_train_steps 10000 --n_steps_per_epoch 2000 --save_freq 1000 --action_weight 0 --loader datasets.SequenceDatasetSpermNormalized --loss_type sperm_loss --renderer utils.EMARenderer --n_diffusion_steps 20 --learning_rate 2e-5 ----data_file diffuser/datasets/BezierSplinesData/inmotile
```

2. Generate spermatozoa trajectories of each type using a real or a gaussian sampled condition.
- Progressive trajectories from real condition
```
python scripts/paper_images/generate_sperms_jsons_rans_larger_real_init_condition.py --dataset SingleSpermBezierIncrementsDataAugSimplified-v0 --logbase logs --diffusion_loadpath diffusion/defaults_H16_T20/progressive_model --policy sampling.DiffPolicy --horizon 16
```
- Progressive trajectories from gaussian condition. Uses data from [BezierSplinesData](https://github.com/SergioHdezG/sperm-diffuser/tree/main/diffuser/datasets/BezierSplinesData) to obtain the initial conditions.
```
python scripts/paper_images/generate_sperms_jsons_simplified_gauss_init.py --dataset SingleSpermBezierIncrementsDataAugSimplified-v0 --logbase logs --diffusion_loadpath diffusion/defaults_H16_T20/progressive_model --policy sampling.DiffPolicy --horizon 16
```
- Slow trajectories from real condition. Use next parameter on ``` generate_sperms_jsons_rans_larger_real_init_condition.py```: ```--diffusion_loadpath diffusion/defaults_H16_T20/progressive_model```
- Slow trajectories from gaussian condition. Use next parameter on ``` generate_sperms_jsons_simplified_gauss_init.py```: ```--diffusion_loadpath diffusion/defaults_H16_T20/progressive_model```
- Inmotile trajectories from real condition. Use next parameter on ``` generate_sperms_jsons_rans_larger_real_init_condition.py```: ```--diffusion_loadpath diffusion/defaults_H16_T20/inmotile_model```
- Inmotile trajectories from gaussian condition. Use next parameter on ``` generate_sperms_jsons_simplified_gauss_init.py```: ```--diffusion_loadpath diffusion/defaults_H16_T20/inmotile_model```

3. Obtain metrics:

Use the next script to obtain KL distance measures:
```
python scripts/measurements/measure_generated_dataset_simplified_model.py 
```

Use the next script to obtain other measures:

```
python scripts/measurements/measure_EMD2traj_generated_dataset_simplified_model.py
```

Note that the paths needs to be set to load the desired dataset:
- **data_file:** Ground truth data ('diffuser/datasets/BezierSplinesData/progressive').
- **synth_data_file:** Synthetic data path ('diffuser/datasets/synthdata_progressive_sperm/progressive_data').
- **figures_path:** path to save figures.

If you use this code or dataset in your research, please cite it as follows:

##  Citation

```bibtex
@article{hernandez2025real,
  title={Real-like synthetic sperm video generation from learned behaviors},
  author={Hern{\'a}ndez-Garc{\'\i}a, Sergio and Cuesta-Infante, Alfredo and Makris, Dimitrios and S. Montemayor, Antonio},
  journal={Applied Intelligence},
  volume={55},
  number={6},
  pages={518},
  year={2025},
  publisher={Springer}
}
