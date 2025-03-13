import torch
import sys
sys.path.append("./model")
sys.path.append("./utils")
import torch.nn as nn
from utils.eval_metrics import evaluate
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from dataset.readdataset import Dydata
import numpy as np
import os
import logging

def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# logger = logger_config(os.path.join(r"log", 'log_own_256_30_10_reid.txt'), "pointnet-lstm")
logger = logger_config(os.path.join(r"log", 'log.txt'), "AOD-MGC")
def logger_info(info):
    # print(info)
    logger.info(info)




batch_size = 32
torch.manual_seed(1234)
rng = torch.Generator()
rng.manual_seed(1234)
text_dataset = Dydata(r"../data_processed3/test")
#text_dataset = Dydata(r"../s")


query_ratio = 0.3
query_size = int(len(text_dataset)*query_ratio)
gallery_size = len(text_dataset)-query_size

query_dataset, gallery_dataset = random_split(text_dataset, [query_size, gallery_size], generator=rng)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

querylaoder = DataLoader(query_dataset, batch_size)
gallerylaoder = DataLoader(gallery_dataset, batch_size)

pointg = torch.load(r"state/model_34.pth")

pointg = pointg.to(device)



for epoch in range(500):
    print("------------------第{}轮测试---------------".format(epoch + 1))

    pointg.eval()
    with torch.no_grad():
        qf, q_pids= [], []
        for data in querylaoder:
            input, label = data
            input = input.to(torch.float)
            input = input.to(device)


            feature= pointg(input, None)

            qf.append(feature)
            q_pids.extend(label)
            
            
    

        qf = torch.cat(qf, 0).cpu()
        qf = np.asarray(qf)
        q_pids = np.asarray(q_pids)

        gf, g_pids = [], []

        for data in gallerylaoder:
            input, label = data
            input = input.to(torch.float)
            input = input.to(device)

            feature = pointg(input, None)

            gf.append(feature)
            g_pids.extend(label)

        gf = torch.cat(gf, 0).cpu()
        gf = np.asarray(gf)
        g_pids = np.asarray(g_pids)

        # 检查 query_features 是否包含 NaN 值
        if np.isnan(qf).any():
            # 处理 NaN 值，例如填充为 0
            qf[np.isnan(qf)] = 0

        # 检查 gallery_features 是否包含 NaN 值
        if np.isnan(gf).any():
            # 处理 NaN 值，例如填充为 0
            gf[np.isnan(gf)] = 0


        top1_acc, top3_acc, topk_acc, mAP = evaluate(qf, gf, q_pids, g_pids)
        print("第{}次测试结果:Top-1 Accuracy:{},Top-3 Accuracy:{} Top-5 Accuracy:{},mAP:{}".format(epoch+1,top1_acc,top3_acc,topk_acc,mAP))



