# Learning Multiphysics Dynamics of Microwave Ablation Using a Multimodal Neural Network

Official PyTorch implementation of **"Learning Multiphysics Dynamics of Microwave Ablation Using a Multimodal Neural Network"** (under review at IEEE Journal of Biomedical and Health Informatics).

Seonaeng Cho, Juil Park, Youjin Kim, Minwoo Shin, Joon Ho Kwon, and Kyungho Yoon

Given a 3D tumor/vessel geometry, the needle position and the ablation settings (time and power), the networks predict the 3D **temperature distribution** and **thermal damage region** of microwave ablation (MWA) in a single forward pass, replacing the numerical simulation.

## Repository structure

```
├── config.py               # all settings: model, paths, hyperparameters, seeds
├── utils.py                # temperature scaling, plotting
├── loss.py                 # training losses and evaluation metrics
├── model/                  # network architectures
│   ├── __init__.py         # build_model(): builds the model selected in config.py
│   ├── AffineUNet1/2/3.py
│   ├── ConcatNet.py
│   ├── CrossAttenUNet8/12/16.py
│   └── ViTEncoder.py
├── Train_Temp.py           # train a temperature model
├── Train_Dmg.py            # train a damage model
├── Test_Temp.py            # evaluate a temperature model
└── Test_Dmg.py             # evaluate a damage model
```

## Installation

```bash
git clone https://github.com/SeonAengCho/mwa-multimodal-multiphysics.git
cd mwa-multimodal-multiphysics
conda create -n mwa python=3.10
conda activate mwa
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data

The tumor and vessel geometries are derived from the hepatic vessel dataset (Task08_HepaticVessel) of the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) [1, 2]. For each tumor, 300 MWA simulations with different needle positions and ablation settings were run. The tumors are split into 36 training, 12 validation and 12 test cases. The simulation data are not included in this repository.

[1] M. Antonelli et al., "The Medical Segmentation Decathlon," *Nature Communications*, vol. 13, 4128, 2022.\
[2] A. L. Simpson et al., "A large annotated medical image dataset for the development and evaluation of segmentation algorithms," arXiv:1902.09063, 2019.

The scripts expect the following files in `Dataset/` (N = number of tumors in the split, 300 samples per tumor):

| File | Shape | dtype | Description |
|---|---|---|---|
| `{split}_input_onlytumor.npy` | (N, 81, 81, 81) | uint8 | tumor mask |
| `{split}_input_vessel.npy` | (N, 81, 81, 81) | uint8 | vessel mask |
| `{split}_input_needle.npy` | (300·N, 6) | float32 | normalized needle coordinates |
| `{split}_input_cond.npy` | (300·N, 2) | float32 | normalized ablation time and power |
| `{split}_label_temp.npy` | (300·N, 81, 81, 81) | float32 | temperature (°C) |
| `{split}_label_dmg.npy` | (300·N, 81, 81, 81) | uint8 | binary damage region |

`{split}` is `train`, `valid` or `test`.

## Configuration

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `temp_model`, `dmg_model` | `{"name": "AffineUNet3"}` | model of each task; the available names and options are listed in `config.py` |
| `data_dir`, `weight_dir`, `result_dir` | `Dataset`, `weight`, `Results` | data, weight and output folders |
| `learning_rate`, `epoch_num`, `batch_size` | 1e-4, 100, 16 | training hyperparameters (both tasks) |
| `train_seed`, `valid_seed` | `None` | `None` draws a random seed each run and prints it; set an integer to reproduce a run |
| `test_seed` | 42 | selects the 1,080 Unforeseen test samples |

`Train_*.py` saves the trained weights as `weight/<model name>.pth` (e.g. `AffineUNet3_temp.pth`), and `Test_*.py` loads the same file. Pretrained weights are not provided.

## Training

```bash
python Train_Temp.py
python Train_Dmg.py
```

Of each training tumor's 300 samples, the first 90% are used for training. The best epoch is selected on 1,080 validation samples (early stopping after 15 epochs without improvement).

## Evaluation

```bash
python Test_Temp.py
python Test_Dmg.py
```

Two settings are reported:
- **Foreseen**: the remaining 10% of samples of the training tumors (unseen needle positions and ablation settings).
- **Unforeseen**: 1,080 samples from the unseen test tumors.

| Task | Metrics |
|---|---|
| Temperature | RMSE, MAE (°C), Dice of the region above 50 °C |
| Damage | Dice, IoU, HD95, ASD (prediction thresholded at 0.5) |

Every metric is computed per sample and averaged. ASD is measured from the ground-truth surface to the predicted surface (Eq. (16) of the paper). Example predictions are saved to `Results/`.

Tensor shapes in the code comments use `B` for the batch size; the volumes are 81×81×81 (resized to 80×80×80 inside the networks).

## Acknowledgements

`model/ViTEncoder.py` is adapted from [tamasino52/UNETR](https://github.com/tamasino52/UNETR) (MIT License).

## Citation

The paper is under review. Citation information will be added after publication.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
