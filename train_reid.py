import torch
import sys
sys.path.append("./model")
sys.path.append("./utils")
import torch.nn as nn
from utils.losses import TripletLoss
from utils.eval_metrics import evaluate
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from dataset.readdataset import Dydata
from model.aod_mgc import pointnet_mgc
import matplotlib.pyplot as plt
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


logger = logger_config(os.path.join(r"log", 'action.txt'), "pointnet-transformer")
def logger_info(info):
    # print(info)
    logger.info(info)


# batch_size = 10
# dataset = Dydata(r"data_qt_processed4")
# torch.manual_seed(1234)
# train_ratio = 10.8
# train_size = int(train_ratio * len(dataset))
# test_size = len(dataset) - train_size
# rng = torch.Generator()
# rng.manual_seed(1234)
#
# train_dataset, text_dataset = random_split(dataset, [train_size, test_size], generator=rng)
#
# query_ratio = 10.5
# query_size = int(len(text_dataset)*query_ratio)
# gallery_size = len(text_dataset)-query_size
#
#
# query_dataset, gallery_dataset = random_split(text_dataset, [query_size, gallery_size], generator=rng)


batch_size = 32
train_dataset = Dydata(r"../data_processed3/train")
torch.manual_seed(1234)
rng = torch.Generator()
rng.manual_seed(1234)
text_dataset = Dydata(r"../data_processed3/train")

query_ratio = 0.5
query_size = int(len(text_dataset)*query_ratio)
gallery_size = len(text_dataset)-query_size

query_dataset, gallery_dataset = random_split(text_dataset, [query_size, gallery_size], generator=rng)

# indices_x = []
# indices_y = []
# i = 0
# j = i + 1
# while i < len(text_dataset):
#     indices_x.append(i)
#     indices_y.append(i + 1)
#     i += 2
#
# query_dataset = Subset(text_dataset, indices_x)
# gallery_dataset = Subset(text_dataset, indices_y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainloader = DataLoader(train_dataset,  batch_size, shuffle=True,drop_last=True)
querylaoder = DataLoader(query_dataset, batch_size)
gallerylaoder = DataLoader(gallery_dataset, batch_size)

pointg = pointnet_mgc(3)

pointg = pointg.to(device)


loss_cro = nn.CrossEntropyLoss()
loss_tri = TripletLoss()
loss_fn = loss_cro.to(device)
loss_tri = loss_tri.to(device)
optim = torch.optim.Adam(pointg.parameters(), 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)


train_step = 0
epoch_list = []
train_list_loss = []
test_list_acc = []
for epoch in range(500):
    print("------------------第{}轮训练---------------".format(epoch + 1))
    pointg.train()
    total_loss = 0
    num = 0
    pred_acc = 0

    epoch_list.append(epoch+1)
    torch.cuda.empty_cache()  # 释放显存


    for data in trainloader:
        with torch.autograd.set_detect_anomaly(True):

            input,label = data
            input = input.to(torch.float)
            input = input.to(device)
            label = label.to(device)
            f = pointg(input, None)

            loss = loss_tri(f, label)

            optim.zero_grad()

            loss.backward()
            optim.step()
            total_loss=total_loss+loss*input.size(0)

            for i in range(len(label)):
                num+=1
            train_step+=1
            print("训练次数{}，当前损失{}".format(train_step, loss))


    # writer.add_scalar("train_loss", total_loss/num, epoch+11)
    scheduler.step()
    # train_list_loss.append(total_loss.detach().cpu().numpy()/num)

    # if (epoch+1)%10==0:
    #
    #     plt.figure()
    #     plt.plot(epoch_list, train_list_loss, label="train_loss")
    #     plt.xlabel("n_epoch")
    #     plt.ylabel("Loss")
    #     plt.title("train_loss")
    #     plt.savefig("pic/train_loss/train_loss_{}.jpg".format(epoch+1))



    pointg.eval()
    test_acc = 0
    t_num = 0
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


        top1_acc, topk_acc, mAP = evaluate(qf, gf, q_pids, g_pids)

        logger_info("第{}次训练结果:Top-1 Accuracy:{},Top-5 Accuracy:{},mAP:{}".format(epoch+1,top1_acc,topk_acc,mAP))


