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
   
First, set up an environment with PyTorch and clone this repository to access our custom code. Here, we refer to nnU-Net v2 repository [installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for a detailed explanation but provide a rundown on how to set up the environment quickly:

#### 1. Create and activate your Conda environment (recommended)
```python
conda create -n panther python=3.9
conda activate panther
```

#### 2. Install [PyTorch](https://pytorch.org/get-started/locally) as described on their website (conda/pip)

#### 3. Clone this repository
```bash
git clone https://github.com/MIC-DKFZ/panther.git
cd panther
```

#### 4. Install nnU-Net v2 and our project in editable mode
This makes our custom trainers and plans available to the framework.
```python
pip install -e .
```

#### 5. Set environment variables
We recommend editing the `.bashrc` (or corresponding startup shell) file of your home folder by adding the following lines:
```python
export nnUNet_raw="/panther/nnUNet_raw"
export nnUNet_preprocessed="/panther/nnUNet_preprocessed"
export nnUNet_results="/panther/nnUNet_results"
```
(Change the paths according to your preferred directory structure.)

### Pre-trained Checkpoints

Coming soon!

### Evaluation

Coming soon!

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