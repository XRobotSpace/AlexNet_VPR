# -*- encoding: utf-8 -*-
'''
@File          :  feature_extractor.py
@Description   :  
@From          :  
@Time          :  2024/05/06 10:28:17
@Author        :  xrobot
@Vision        :  1.0
'''


from abc import ABC, abstractmethod
import numpy as np
from typing import List


class FeatureExtractor(ABC):

    @abstractmethod
    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        pass