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
[2024-12-05] Our paper is available online! [https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2024.124878] 

##Abstract
The goal of infrared and visible image fusion (IVIF) is to integrate the unique advantages of both modalities to achieve a more comprehensive understanding of a scene. However, existing methods struggle to effectively handle modal disparities, resulting in visual degradation of the details and prominent targets of the fused images. To address these challenges, we introduce PromptFusion, a prompt-based approach that harmoniously combines multi-modality images under the guidance of semantic prompts. Firstly, to better characterize the features of different modalities, a contourlet autoencoder is designed to separate and extract the high-/low-frequency components of different modalities, thereby improving the extraction of fine details and textures. We also introduce a prompt learning mechanism using positive and negative prompts, leveraging Vision-Language Models to improve the fusion model’s understanding and identification of targets in multi-modality images, leading to improved performance in downstream tasks. Furthermore, we employ bi-level asymptotic convergence optimization. This approach simplifies the intricate non-singleton non-convex bi-level problem into a series of convergent and differentiable single optimization problems that can be effectively resolved through gradient descent. Our approach advances the state-of-the-art, delivering superior fusion quality and boosting the performance of related downstream tasks.

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
