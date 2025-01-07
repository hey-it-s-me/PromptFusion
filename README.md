# PromptFusion: Harmonized Semantic Prompt Learning for Infrared and Visible Image Fusion


[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8.1-%237732a8)](https://pytorch.org/)



### PromptFusion: Harmonized Semantic Prompt Learning for Infrared and Visible Image Fusion [IEEE/CAA JAS 2024]

By Jinyuan Liu, Xingyuan Li, Zirui Wang, Zhiying Jiang, Wei Zhong, Wei Fan and Bin Xu

<div align=center>
<img src="https://github.com/hey-it-s-me/PromptFusion/blob/main/network.png" width="50%">
</div>

## Updates
[2024-12-05] Our paper is available online! [[Paper](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2024.124878)] 

## Requirements
- CUDA 11.1
- Python=3.8.10
- Pytorch 1.8.1
## Training
1. Environment Preparation
```
# create virtual environment
conda create -n promptfusion python=3.8.10
conda activate promptfusion
# install requirements
pip install -r requirements.txt
```
2. Data Preparation
   
We use [MSRS](https://github.com/Linfeng-Tang/MSRS) dataset for training, and [M3FD](https://github.com/JinyuanLiu-CV/TarDAL), [TNO](http://figshare.com/articles/TNO\_Image\_Fusion\_Dataset/1008029), [RoadScene](https://github.com/hanna-xu/RoadScene) datasets for evalution.
   
Run
```
python dataprocessing.py
```
to process the data before training.

3. Start training
   
Run the training code to get started.
```
python train.py
```
## Experiments results
## Citation
```
@article{liu2024promptfusion,
  title={PromptFusion: Harmonized Semantic Prompt Learning for Infrared and Visible Image Fusion},
  author={Liu, Jinyuan and Li, Xingyuan and Wang, Zirui and Jiang, Zhiying and Zhong, Wei and Fan, Wei and Xu, Bin},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={12},
  pages={1--14},
  year={2024},
  publisher={IEEE/CAA Journal of Automatica Sinica}
}
```
## Acknowledgement
Our codes are based on [CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse), [Contourlet-CNN](https://github.com/xKHUNx/Contourlet-CNN), thanks for their contribution.
