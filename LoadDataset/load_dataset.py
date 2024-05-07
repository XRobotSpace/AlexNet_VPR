# -*- encoding: utf-8 -*-
'''
@File          :  load_dataset.py
@Description   :  
@From          :  https://github.com/stschubert/VPR_Tutorial
@Time          :  2024/04/29 16:27:19
@Author        :  xrobot
@Vision        :  1.0
'''


import os
import urllib.request
import zipfile
from glob import glob
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from typing import List, Tuple
from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    返回 参考集合-imgs_db, 查询集合-imgs_q, Hard 差异矩阵-GThard, Soft 差异矩阵-GTsoft
    """
    @abstractmethod
    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def download(self, destination: str):
        pass


class GardensPointDataset(Dataset):
    def __init__(self, destination: str = 'Datasets/GardensPoint/'):
        self.destination = destination  # 数据集保存目录

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset GardensPoint day_right--night_right')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + 'day_right/*.jpg'))
        fns_q = sorted(glob(self.destination + 'night_right/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        GThard = np.eye(len(imgs_db)).astype('bool')
        GTsoft = convolve2d(GThard.astype('int'),
                            np.ones((17, 1), 'int'), mode='same').astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== GardensPoint dataset does not exist. Download to ' + destination + '...')

        fn = 'GardensPoint_Walking.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)


class StLuciaDataset(Dataset):
    def __init__(self, destination: str = 'Datasets/StLucia_small/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset StLucia 100909_0845--180809_1545 (small version)')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + '100909_0845/*.jpg'))
        fns_q = sorted(glob(self.destination + '180809_1545/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        gt_data = np.load(self.destination + 'GT.npz')
        GThard = gt_data['GThard'].astype('bool')
        GTsoft = gt_data['GTsoft'].astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== StLucia dataset does not exist. Download to ' + destination + '...')

        fn = 'StLucia_small.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)


class SFUDataset(Dataset):
    def __init__(self, destination: str = 'Datasets/SFU/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset SFU dry--jan')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + 'dry/*.jpg'))
        fns_q = sorted(glob(self.destination + 'jan/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        gt_data = np.load(self.destination + 'GT.npz')
        GThard = gt_data['GThard'].astype('bool')
        GTsoft = gt_data['GTsoft'].astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== SFU dataset does not exist. Download to ' + destination + '...')

        fn = 'SFU.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)
