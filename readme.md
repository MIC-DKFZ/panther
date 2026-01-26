# [MICCAI2025 Panther] Multi-Stage Fine-Tuning and Ensembling for Pancreatic Tumor Segmentation in MRI

This repository contains the official implementation of our MICCAI 2025 challenge paper:

### **A Multi-Stage Fine-Tuning and Ensembling Strategy for Pancreatic Tumor Segmentation in Diagnostic and Therapeutic MRI**

We present a **cascaded pre-training, fine-tuning, and ensembling framework** for **pancreatic ductal adenocarcinoma (PDAC) segmentation** in MRI. Our method, built on **nnU-Net v2**, leverages:  
- Multi-stage cascaded fine-tuning (from a foundation model → CT lesions → target MRI modalities)  
- Systematic augmentation ablations (aggressive vs. default)  
- Metric-aware heterogeneous ensembling ("mix of experts")  

This approach achieved **the first place** in the **PANTHER challenge**, with robust performance across both **diagnostic T1W (Task 1)** and **therapeutic T2W MR-Linac (Task 2)** scans.

> **Authors**: Omer Faruk Durugol*, Maximilian Rokuss*, Yannick Kirchhoff, Klaus H. Maier-Hein  
> *Equal contribution  
> **Paper**: [![arXiv](https://img.shields.io/badge/arXiv-2508.21775-b31b1b.svg)](https://arxiv.org/abs/2508.21775)  
> **Challenge**: [PANTHER](https://panther.grand-challenge.org)

---

## News/Updates
- 🏆 **Aug 2025**: Achieved **1st place in both tasks** in the [PANTHER Challenge](https://panther.grand-challenge.org)!  
- 📄 **Aug 2025**: Paper preprint released on [arXiv](https://arxiv.org/abs/2508.21775).  

---

## 🚀 Usage

This section provides a complete guide to reproduce our training and inference workflow. Our method is developed entirely within the nnU-Net v2 framework.

### Installation and Setup
   
We need to set up an environment with PyTorch and clone this repository to access our custom code. Here, we refer to nnU-Netv2 repository [installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for a detailed explanation but provide a rundown on how to set up the environment quickly:

#### 1. Create and activate your Conda environment (recommended)
```python
conda create -n panther python=3.10 # A python version >=3.10 is needed
conda activate panther
```

#### 2. Install [PyTorch](https://pytorch.org/get-started/locally) as described on their website (conda/pip)

#### 3. Clone this repository
```bash
git clone https://github.com/MIC-DKFZ/panther.git
cd panther
```

#### 4. Install nnUNet v2 and our project in editable mode
This makes our custom trainers and plans available to the framework.
```python
pip install -e .
```

#### 5. Set environment variables
For the nnUNetv2 training commands to work properly, certain environment variables need to be set up. These are the folder paths for the raw unprocessed datasets, the preprocessed datasets and the trained checkpoints (respectively `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`). We recommend editing the `.bashrc` (or your corresponding terminal shell, e.g. `.zshrc` for zsh shell) file of your home folder by adding the lines specified in "Method 1" below. You can also opt for "Method 2" to export the environment variables only in the terminal you're using for running nnUNetv2 commands, however beware that the environment variables would be only valid in the current terminal they are set, as they do not persist between terminals, and any terminal if closed, need the environment variables to be set up again. We also include the code for Windows OS to set up persistent or current-terminal-only environment variables.

Method 1: If you're editing `.bashrc` (change this according to your shell e.g. `.zshrc`) for persistent terminals with environment variables set up automatically when a terminal opens:
```python
cat << 'EOF' >> ~/.bashrc
# nnUNet environment variables
export nnUNet_raw="/panther/nnUNet_raw"
export nnUNet_preprocessed="/panther/nnUNet_preprocessed"
export nnUNet_results="/panther/nnUNet_results"
EOF
```
(You can change the paths according to your preferred directory structure.)
For Windows OS, open a command line (cmd) and paste below to set environment variables across all cmd and PowerShell terminals (except current one, so you have to restart your terminal):
```python
setx nnUNet_raw="C:\panther\nnUNet_raw"
setx nnUNet_preprocessed="C:\panther\nnUNet_preprocessed"
setx nnUNet_results="C:\panther\nnUNet_results"
```

Method 2: If you don't want to change the shell source, paste the variables below whenever opening up a new terminal:
```python
export nnUNet_raw="/panther/nnUNet_raw"
export nnUNet_preprocessed="/panther/nnUNet_preprocessed"
export nnUNet_results="/panther/nnUNet_results"
```
(You can change the paths according to your preferred directory structure.)
For Windows OS, open a command line (cmd) and paste below to set environment variables for the current command line (if you close the terminal, you need to set up the environment variables again):
```python
set nnUNet_raw="C:\panther\nnUNet_raw"
set nnUNet_preprocessed="C:\panther\nnUNet_preprocessed"
set nnUNet_results="C:\panther\nnUNet_results"
```

### Run Inference for Task 1

#### 1. Download the provided [Task 1 model folds](https://zenodo.org/records/17212158)

#### 2. Place the model folder `Dataset901_PANTHER` in the `nnUNet_results` folder (default location at path `panther/nnUNet_results`) 

#### 3. Run the command below replacing your input folder (-i) and output folder (-o).
```python
nnUNetv2_predict_from_modelfolder \
-i path/to/input/folder \
-o path/to/output/folder \
-m $nnUNet_results/Dataset901_PANTHER/nnUNetTrainer1e3__nnUNetResEncUNetLPlansPancCTMultiTalentV2__3d_fullres_iso1x1x1 \
-f (0,1,2,3,4,5) \
-chk checkpoint_best.pth
```

#### 4. Your segmentation results will be in the output folder.

### Run Inference for Task 2

#### 1. Download the provided [Task 2 model folds](https://zenodo.org/records/17212158)

#### 2. Place the model folder `Dataset902_PANTHER_HR` in the `nnUNet_results` folder (default location at path `panther/nnUNet_results`)

#### 3. Run the command below replacing your input folder (-i) and output folder (-o).
```python
nnUNetv2_predict_from_modelfolder \
-i path/to/input/folder \
-o path/to/output/folder \
-m $nnUNet_results/Dataset902_PANTHER_HR/nnUNetTrainer1e3__nnUNetResEncUNetLPlansTask1PancCTMultiTalentV2__3d_fullres_iso1x1x1 \
-f (0,1,2,3,4) \
-chk checkpoint_best.pth
```

#### 4. Your segmentation results will be in the output folder.

### Finetuning from Pretrained Checkpoints

Our method for PANTHER dataset utilizes the checkpoint [PancCTMultiTalentV2](https://zenodo.org/records/17212158), which is initialized from the [MultiTalentV2](https://zenodo.org/records/13753413) weights, and pretrained on 765 pancreas tumor cases from [MSD pancreas](http://medicaldecathlon.com/dataaws/) and [PANORAMA](https://panorama.grand-challenge.org/datasets-imaging-labels/) datasets. ([MultiTalentV2](https://zenodo.org/records/13753413) is pretrained over 40 MRI/PET/CT abdominal organ segmentation datasets.)

#### 1. Download the [PancCTMultiTalentV2](https://zenodo.org/records/17212158) checkpoint

#### 2. Place the model folder `Dataset999_PancCTPretrain` in the `nnUNet_results` folder (located by default at path `panther/nnUNet_results`)

#### 3. Change the below training code to match your 2 or 3 digits (xxx) dataset number from `Datasetxxx_DATASETNAME` naming convention, (Your dataset must be preprocessed! If you haven't preprocessed your dataset, refer to Preprocessing Your Dataset section.), the plans json file name, the chosen configuration from those defined in the plans json file, and the pretrained checkpoint path if different from default. The training code template is as:
```python
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIG FOLD
```
Thus to run a single fold, run (changing `xxx` to your dataset number):
```python
nnUNetv2_train xxx 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer1e3 -pretrained_weights $nnUNet_results/Dataset999_PancCTPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_best.pth
```
To run 5-fold cross-validation on multiple GPUs at once, run the following code block:
```python
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train xxx 3d_fullres_iso1x1x1 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer1e3 -pretrained_weights $nnUNet_results/Dataset999_PancCTPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_best.pth & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train xxx 3d_fullres_iso1x1x1 1 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer1e3 -pretrained_weights $nnUNet_results/Dataset999_PancCTPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_best.pth & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train xxx 3d_fullres_iso1x1x1 2 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer1e3 -pretrained_weights $nnUNet_results/Dataset999_PancCTPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_best.pth & # train on GPU 2
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train xxx 3d_fullres_iso1x1x1 3 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer1e3 -pretrained_weights $nnUNet_results/Dataset999_PancCTPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_best.pth & # train on GPU 3
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train xxx 3d_fullres_iso1x1x1 4 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer1e3 -pretrained_weights $nnUNet_results/Dataset999_PancCTPretrain/nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4/fold_all/checkpoint_best.pth & # train on GPU 4
```

### Preprocessing Your Dataset (--Not complete, in progress--)

nnUNetv2 offers automatic data preprocessing. While we could you could do preprocessing at one step with `nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity`, you would have to m we recommend using the preprocessing steps separately (in the order shown below) 

#### 1. Extract dataset fingerprint

Run the below command (change `xxx` to your dataset id):
```python
nnUNetv2 nnUNetv2_extract_fingerprint -d xxx 
```

#### 2. Plan experiment

To use ResEnc-L encoder, we need to specify that in the plan experiment phase via argument -pl, you can change the argument to `nnUNetPlannerResEncX` with X being any option from (M/L/XL) to choose your ResEnc encoder. Then, run the below command:
```python
nnUNetv2_plan_experiment -d xxx -pl nnUNetPlannerResEncL
```

#### 3. Preprocess the dataset
```python
nnUNetv2_preprocess -d xxx -c 3d_fullres_iso1x1x1 -p nnUNetResEncUNetLPlansPancCTMultiTalentV2
```

Rest is coming soon!

### Evaluation on PANTHER Metrics (--Not complete, in progress--)

Five different metrics are used by the PANTHER Challenge to rank the final submissions, only on the tumor labels, these are: Dice, 5mm Surface Dice, MASD (Mean Average Surface Distance), HD95 (Hausddorff Distance @ 95th Percentile), RMSE (Root Mean Squared Error). All of the metrics in the official evaluation is implemented from the DeepMind's `surface-distance` package.

Rest is coming soon!

---

## 📂 Data

We trained and evaluated on the **PANTHER Challenge dataset**:
👉 [https://zenodo.org/records/15192302](https://zenodo.org/records/15192302)

Pretraining leveraged:

* [MultiTalentV2](https://zenodo.org/records/13753413)
* Pancreatic lesion CT datasets (MSD + PANORAMA)

---

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@article{durugol2025multistagefinetuningensemblingstrategy,
      title={A Multi-Stage Fine-Tuning and Ensembling Strategy for Pancreatic Tumor Segmentation in Diagnostic and Therapeutic MRI}, 
      author={Omer Faruk Durugol and Maximilian Rokuss and Yannick Kirchhoff and Klaus H. Maier-Hein},
      year={2025},
      eprint={2508.21775},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.21775}, 
}
```

---

## 📬 Contact

For questions, issues, or collaborations, please reach out:
📧 [maximilian.rokuss@dkfz-heidelberg.de](mailto:maximilian.rokuss@dkfz-heidelberg.de)