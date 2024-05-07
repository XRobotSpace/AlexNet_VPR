# -*- encoding: utf-8 -*-
'''
@File          :  show_correct_and_wrong_matches.py
@Description   :  
@From          :  
@Time          :  2024/05/07 09:35:53
@Author        :  xrobot
@Vision        :  1.0
'''


from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from typing import Tuple, List, Optional


def add_frame(img_in: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Adds a colored frame around an input image.

    Args:
        img_in (np.ndarray): A three-dimensional array representing the input image (height, width, channels).
        color (Tuple[int, int, int]): A tuple of three integers representing the RGB color of the frame.

    Returns:
        np.ndarray: A three-dimensional array representing the image with the added frame.
    """    
    img = img_in.copy()

    w = int(np.round(0.01 * img.shape[1]))

    # pad left-right
    pad_lr = np.tile(np.uint8(color).reshape(1, 1, 3), (img.shape[0], w, 1))
    img = np.concatenate([pad_lr, img, pad_lr], axis=1)

    # pad top-bottom
    pad_tb = np.tile(np.uint8(color).reshape(1, 1, 3), (w, img.shape[1], 1))
    img = np.concatenate([pad_tb, img, pad_tb], axis=0)

    return img


def show(db_imgs: List[np.ndarray],
         q_imgs: List[np.ndarray],
         TP: np.ndarray,
         FP: np.ndarray,
         M: Optional[np.ndarray] = None):
    """
    Displays a visual comparison of true positive and false positive image pairs
    from a database and query set. Optionally, a similarity matrix can be included.

    Args:
        db_imgs (List[np.ndarray]): A list of 3D arrays representing the database images (height, width, channels).
        q_imgs (List[np.ndarray]): A list of 3D arrays representing the query images (height, width, channels).
        TP (np.ndarray): A two-dimensional array containing the indices of true positive pairs.
        FP (np.ndarray): A two-dimensional array containing the indices of false positive pairs.
        M (Optional[np.ndarray], optional): A two-dimensional array representing the similarity matrix. Defaults to None.

    Returns:
        None: This function displays the comparison result using matplotlib.pyplot but does not return any value.
    """
    # true positive TP
    if(len(TP) == 0):
        print('No true positives found.')
        return
    print(TP)
    idx_tp = np.random.permutation(len(TP))[:1]     # 比如 TP 中有 8 组数据，将 0-7 随机打乱，选择第一个数，即行号
    
    db_tp = db_imgs[int(TP[idx_tp, 0])]     # 根据 DB 中真实匹配的图像索引，获取图像
    q_tp = q_imgs[int(TP[idx_tp, 1])]       # 根据 Q 中对应的图像索引，获取图像
    
    # 如果形状不匹配，就会调整查询图像的大小，使其与数据库图像的形状相匹配
    if db_tp.shape != q_tp.shape:
        q_tp = resize(q_tp.copy(), db_tp.shape, anti_aliasing=True)
        q_tp = np.uint8(q_tp * 255)   # 将像素值缩放到 0 到 255 的范围内
        
    img = add_frame(np.concatenate([db_tp, q_tp], axis=1), [119, 172, 48])


    # false positive FP
    try:
        idx_fp = np.random.permutation(len(FP))[:1]

        db_fp = db_imgs[int(FP[idx_fp, 0])]
        q_fp = q_imgs[int(FP[idx_fp, 1])]

        if db_fp.shape != q_fp.shape:
            q_fp = resize(q_fp.copy(), db_fp.shape, anti_aliasing=True)
            q_fp = np.uint8(q_fp*255)

        img_fp = add_frame(np.concatenate([db_fp, q_fp], axis=1), [162, 20, 47])
        img = np.concatenate([img, img_fp], axis=0)
    except:
        pass

    # concat M
    if M is not None:
        M = resize(M.copy(), (img.shape[0], img.shape[0]))
        M = np.uint8(M.astype('float32')*255)
        M = np.tile(np.expand_dims(M, -1), (1, 1, 3))
        img = np.concatenate([M, img], axis=1)
    
    return img
