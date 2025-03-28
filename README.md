# DVSRNet

Code of the paper "DVSRNet: Deep Video Super-Resolution Based on Progressive Deformable Alignment and Temporal-Sparse Enhancement".

# Requirements

CUDA==11.6 Python==3.7 Pytorch==1.13

## Environment
```python
conda create -n DVSRNet python=3.7 -y && conda activate DVSRNet

git clone --depth=1 https://github.com/QZ1-boy/DVSRNet && cd QZ1-boy/DVSRNet/

# given CUDA 11.6
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

## Dataset Download
VSR Training Dataset:

[MMCNN dataset] [MMCNN](https://ieeexplore.ieee.org/document/8579237)

VSR Testing Datasets:

[Myanmar dataset] [Myanmar](https://ieeexplore.ieee.org/document/7444187),
[YUV21 dataset] [YUV21](https://ieeexplore.ieee.org/document/7858640),
[Vid4 dataset] [Vid4](https://ieeexplore.ieee.org/document/8099787)

Optical Flow Training and Testing Datasets:

[Sintel dataset] [Sintel](https://link.springer.com/chapter/10.1007/978-3-642-33783-3_44),
[KITTI dataset] [KITTI](https://ieeexplore.ieee.org/document/7298925),
[FlyingChairs dataset] [FlyChairs](https://ieeexplore.ieee.org/document/7410673),
[FlyingThings3D dataset] [FlyThings3D](https://ieeexplore.ieee.org/document/7780807)


# Train
```python
python train_x2.py
```
# Test
```python
python test_x2.py 
```
# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2024dvsrnet,
  title={Dvsrnet: Deep video super-resolution based on progressive deformable alignment and temporal-sparse enhancement},
  author={Zhu, Qiang and Chen, Feiyu and Zhu, Shuyuan and Liu, Yu and Zhou, Xue and Xiong, Ruiqin and Zeng, Bing},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```
