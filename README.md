ITFPose

mainresult:models can be downloaded from 链接: https://pan.baidu.com/s/13JR_iN43df1jfCdff2wl7Q?pwd=5a3j 提取码: 5a3j

Requirements
Installation

Linux, CUDA>=9.2, GCC>=5.4
Python>=3.6
PyTorch>=1.5.0, torchvision>=0.6.0
mmcv
cd mmcv
pip install -r requirements.txt
pip install -v -e .
mmpose
cd ..
pip install -r requirements.txt
pip install -v -e .

Usage
Dataset preparation
Please download the dataset from https://github.com/AlexTheBad/AP-10K.
CLIP Pretrained models
please download the CLIP Pretrained models from https://github.com/openai/CLIP

training
python tools/train.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ITFPose_ap10k_256x256.py 
Evaluation
python tools/test.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ITFPose_ap10k_256x256.py  checkpionts/epoch_460.pth



