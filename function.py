import numpy as np
import torch
import time
from torch.autograd import Variable


def compute_top5(target, output):

    y_resize = target.view(-1, 1)
    correct_top5 = np.zeros(5)

    for maxk in range(1, 6):
        _, pred = output.topk(maxk, 1)
        correct_top5[maxk - 1] = torch.eq(pred, y_resize).sum().float().item()

    return correct_top5


def train_model(trainLoader, model, epoch, optimizer, criterion, cfg):
    model.train()
    # 损失及正确率累加变量初始化
    running_loss = 0
    running_correct = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(trainLoader):
        # 针对Cuda进行设置
        data, target = data.cuda(), target.cuda()
        # 训练阶段优化器梯度初始化
        optimizer.zero_grad()
        # 模型前向传播
        output = model(data)
        # 得到训练误差
        loss = criterion(output, target)
        # 记录误差及准确率
        running_loss += loss.data
        # running_correct_top5 += compute_top5(target, output)

        pred = output.data.max(1, keepdim=True)[1]  # 获得得分最高的类别
        running_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # 在训练过程，误差反向传播，网络权重参数更新
        loss.backward()
        optimizer.step()

        if batch_idx % cfg.SHOW_STEP == 0:
            print('batch %d, loss is %.4f' % (batch_idx, loss.data.item()))

    # 误差以及准确率计算
    L = 1.0 * len(trainLoader.dataset)
    loss = running_loss.data.item() * cfg.BATCH_SIZE / L
    accuracy = 100.0 * running_correct.data.item() / L
    end_time = time.time()
    print(
        'epoch %d , train loss is %5.4f and train accuracy is %d/%d = %.2f, consume time is %0.2f min'
        % (epoch, loss, running_correct, L, accuracy,
           (end_time - start_time) / 60))
    return accuracy, loss


def valid_model(validLoader, model, epoch, criterion, cfg):
    model.eval()
    # 损失及正确率累加变量初始化
    running_loss = 0.0
    running_correct = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validLoader):
            # 针对Cuda进行设置
            data, target = data.cuda(), target.cuda()

            # 模型前向传播
            output = model(data)
            # 得到训练误差
            loss = criterion(output, target)
            # 记录误差及准确率
            running_loss += loss.data
            # 记录误差及准确率
            pred = output.data.max(1, keepdim=True)[1]  # 获得得分最高的类别
            running_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if batch_idx % cfg.SHOW_STEP == 0:
                print('batch %d, loss is %.4f' % (batch_idx, loss.data.item()))

    # 误差以及准确率计算
    L = 1.0 * len(validLoader.dataset)
    loss = running_loss.data.item() * cfg.BATCH_SIZE / L
    accuracy = 100.0 * running_correct.data.item() / L
    end_time = time.time()
    print(
        'epoch %d , test loss is %5.4f and test accuracy is %d/%d = %.2f, time consume is %.2f min'
        % (epoch, loss, running_correct, L, accuracy,
           (end_time - start_time) / 60))
    return accuracy, loss