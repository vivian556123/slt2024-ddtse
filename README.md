# 
<br>
<p align="center">
<h1 align="center"><strong>DDTSE: DISCRIMINATIVE DIFFUSION MODEL FOR TARGET SPEECH EXTRACTION
</strong></h1>
  </p>

<p align="center">
  <a href="https://arxiv.org/abs/2309.13874" target='_**blank**'>
    <img src="https://img.shields.io/badge/arxiv-2309-13874-blue?">
  </a> 
  <a href="https://vivian556123.github.io/slt2024-ddtse/" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-&#x1f917-blue">
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=OpenRobotLab.pointllm&left_color=gray&right_color=blue">
  </a>
</p>


## 🏠 Introduction

We introduce <b>DDTSE</b>: Discriminative Diffusion Model for Target Speech Extraction and Speech Enhancement. We apply the same forward process as diffusion models and utilize the reconstruction loss similar to discriminative methods. Furthermore, we devise a two-stage training strategy to emulate the inference process during model training.  DDTSE not only works as a standalone system, but also can further improve the performance of discriminative models without additional retraining. Experimental results demonstrate that DDTSE not only achieves higher perceptual quality but also accelerates the inference process by 3 times compared to the conventional diffusion model. 

Please do not hesitate to tell us if you have any feedback!

## 📋 Contents
- [](#)
  - [🏠 Introduction](#-introduction)
  - [📋 Contents](#-contents)
  - [💬 Environment Setup](#-environment-setup)
  - [🔍 Data preparation](#-data-preparation)
  - [📦 Training](#-training)
  - [🤖 Inference:](#-inference)
  - [⛺ Scoring](#-scoring)
  - [🔗 Citation](#-citation)


## 💬 Environment Setup

Create a new virtual environment with Python 3.8 

Install the package dependencies via `pip install -r requirements.txt`.

## 🔍 Data preparation

Please make sure that you have downloaded Libri2Mix. If not, please refer to https://github.com/JorisCos/LibriMix and create your own Libri2Mix dataset. 

## 📦 Training

Training is done by executing `train.py`. 
`bash
python train.py --base_dir <your_base_dir>
`

To run DDTSE for the first stage, please run 
`bash training_command/stage1.sh
`

To run DDTSE for the second stage, please run 
`bash training_command/stage2.sh
`

## 🤖 Inference:

To run DDTSE inference of multi-speaker noisy scenario for the first stage, please run 

`bash inference_command/stage1.sh
`

To run DDTSE inference of multi-speaker noisy scenario for the second stage, please run 

`bash inference_command/stage2.sh
`


## ⛺ Scoring

To evaluate the model performance, please run 

` python calc_metrics.py --gt_dir /directory_or_original_samples  --enhanced_dir /directory_or_generated_samples 
`

## 🔗 Citation

To cite this repository

```bibtex
@article{zhang2024ddtse,
  title={DDTSE: Discriminative Diffusion Model for Target Speech Extraction},
  author={Leying Zhang, Yao Qian, Linfeng Yu, Heming Wang, Hemin Yang, Shujie Liu, Long Zhou, Yanmin Qian},
  journal={IEEE Spoken Language Technology Workshop 2024},
  year={2024}
}
```