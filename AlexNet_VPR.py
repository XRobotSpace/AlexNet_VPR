# -*- encoding: utf-8 -*-
'''
@File          :  AlexNet_VPR.py
@Description   :  
@From          :  
@Time          :  2024/04/29 16:22:01
@Author        :  xrobot
@Vision        :  1.0
'''


import numpy as np
import argparse
from matplotlib import pyplot as plt

from LoadDataset.load_dataset import GardensPointDataset, StLuciaDataset, SFUDataset
from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
from matching import matching
from evaluation import show_correct_and_wrong_matches
from evaluation.metrics import createPR, recallAt100precision, recallAtK


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='Visual Place Recognition')
    parser.add_argument('--descriptor', type=str, default='AlexNet', choices=['HDC-DELF', 'AlexNet', 'NetVLAD', 'PatchNetVLAD', 'CosPlace', 'EigenPlaces', 'SAD'], help='Select descriptor (default: AlexNet)')
    parser.add_argument('--dataset', type=str, default='GardensPoint', choices=['GardensPoint', 'StLucia', 'SFU'], help='Select dataset (default: GardensPoint)')
    args = parser.parse_args()

    print('========== Start VPR with {} descriptor on dataset {}'.format(args.descriptor, args.dataset))    
    
    # 加载数据集：返回参考集合、查询集合、Hard 差异矩阵、Soft 差异矩阵
    print('===== Load dataset')
    if args.dataset == 'GardensPoint':
        dataset = GardensPointDataset()
    elif args.dataset == 'StLucia':
        dataset = StLuciaDataset()
    elif args.dataset == 'SFU':
        dataset = SFUDataset()
    else:
        raise ValueError('Unknown dataset: ' + args.dataset)  
    
    imgs_db, imgs_q, GThard, GTsoft = dataset.load() 
    
    # 定义特征提取器
    if args.descriptor == 'AlexNet':
        feature_extractor = AlexNetConv3Extractor()
    else:
        raise ValueError('Unknown descriptor: ' + args.descriptor)
    
    # 提取特征
    if args.descriptor != 'PatchNetVLAD' and args.descriptor != 'SAD':
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)     # 参考集描述子
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)       # 查询集描述子

        # normalize descriptors and compute S-matrix
        print('===== Compute cosine similarities S')
        db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic , axis=1, keepdims=True)   # 规范化 - 参考集描述子
        q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic , axis=1, keepdims=True)      # 规范化 - 查询集描述子
        S = np.matmul(db_D_holistic , q_D_holistic.transpose())     # 计算相似性矩阵，cosine 距离

    # 匹配决策
    print('===== Match images')
    
    # best match per query -> 单最优匹配
    M1 = matching.best_match_per_query(S)    # 相似度矩阵 S 每列最大的表示为 True，其余为 False
    
    # thresholding -> Multi-match VPR
    M2 = matching.thresholding(S, 'auto')    # 相似度矩阵 S 中大于阈值的标记为 True，其他标记 False    
    
    TP = np.argwhere(M2 & GThard)  # true positives
    FP = np.argwhere(M2 & ~GTsoft)  # false positives    
   
    # evaluation
    print('===== Evaluation')
    # show correct and wrong image matches
    example_img = show_correct_and_wrong_matches.show(imgs_db, imgs_q, TP, FP)  # show random matches   
   
    # PR-curve
    P, R = createPR(S, GThard, GTsoft, matching='multi', n_thresh=100)   
   
    # area under curve (AUC)
    AUC = np.trapz(P, R)
    print(f'\n===== AUC (area under curve): {AUC:.3f}')   
   
    # maximum recall at 100% precision
    maxR = recallAt100precision(S, GThard, GTsoft, matching='multi', n_thresh=100)
    print(f'\n===== R@100P (maximum recall at 100% precision): {maxR:.2f}')   
   
    # recall at K
    RatK = {}
    for K in [1, 5, 10]:
        RatK[K] = recallAtK(S, GThard, GTsoft, K=K)

    print(f'\n===== recall@K (R@K) -- R@1: {RatK[1]:.3f}, R@5: {RatK[5]:.3f}, R@10: {RatK[10]:.3f}')
       
    # 绘制相似度矩阵
    fig = plt.figure()
    plt.imshow(S)
    plt.axis('off')
    plt.title('Similarity matrix S')
    
    # 绘制 M1：相似度矩阵 S 每列最大的表示为 True，其余为 False
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(M1)
    ax1.axis('off')
    ax1.set_title('Best match per query')
    
    # 绘制 M2：相似度矩阵 S 中大于阈值的标记为 True，其他标记 False    
    ax2 = fig.add_subplot(122)
    ax2.imshow(M2)
    ax2.axis('off')
    ax2.set_title('Thresholding S>=thresh')    
    
    # 随机显示正确匹配和错误匹配的样例   
    plt.figure()
    plt.imshow(example_img)
    plt.axis('off')
    plt.title('Examples for correct and wrong matches from S>=thresh')       
    
    # 绘制 PR 曲线  
    plt.figure()
    plt.plot(R, P)
    plt.xlim(0, 1), plt.ylim(0, 1.01)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Result on GardensPoint day_right--night_right')
    plt.grid('on')
    plt.draw()         

    plt.show()


if __name__ == "__main__":
    main()
    