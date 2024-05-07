# -*- encoding: utf-8 -*-
'''
@File          :  feature_extractor.py_holistic.py
@Description   :  
@From          :  https://github.com/stschubert/VPR_Tutorial
@Time          :  2024/05/06 10:38:21
@Author        :  xrobot
@Vision        :  1.0
'''


import numpy as np
from typing import List
import torch
from torchvision import transforms
import torchvision.models as models
from torchvision.models import AlexNet_Weights

from .feature_extractor import FeatureExtractor


class AlexNetConv3Extractor(FeatureExtractor):
    def __init__(self, nDims: int = 4096):
        self.nDims = nDims  # 使用随机投影矩阵降维后的维度
        
        # load AlexNet model
        self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        
        # select conv3
        self.alex_conv3 = self.model.features[:7]
        
        # preprocess images
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # gpu
        gpuid = 0
        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
        
        self.alex_conv3.to(self.device)
        
        
    def compute_features(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        imgs_torch = [self.preprocess(img) for img in imgs]
        imgs_torch = torch.stack(imgs_torch, dim=0).to(self.device)
        
        with torch.no_grad():
            self.alex_conv3.eval()
            output = self.alex_conv3(imgs_torch)
        output = output.to('cpu').numpy()       # 200 * C * W * H
        
        Ds = output.reshape([len(imgs), -1])    # 展平 200 * X
        
        rng = np.random.default_rng(seed=0)
        Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')    # 随机投影矩阵
        Proj = Proj / np.linalg.norm(Proj , axis=1, keepdims=True)  # 每行归一化
        Ds = Ds @ Proj
        return Ds
