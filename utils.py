import numpy as np
import torch
import os
import logging

torch.backends.cudnn.benchmark = True


def train(data_loader, model, criterion, optimizer, use_cuda=True):
    '''
    模型训练
    :param data_loader:数据集
    :param model: 模型
    :param criterion: 评判标准
    :param optimizer: 优化器
    :param use_cuda:是否使用gpu
    :return: 平均损失
    '''
    # 切换模型为训练模式
    model.train()
    # 定义用于保存损失的列表
    train_loss = []
    for i, (input, label) in enumerate(data_loader):
        if use_cuda:  # 如果使用gpu训练 将tensor放到gpu
            input = input.cuda()
            label = label.cuda()

        # y_hat, _ = model(input)  # 获取预测值
        y_hat = model(input)  # 获取预测值
        loss = criterion(y_hat, label)  # 计算损失

        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数
        train_loss.append(loss.item())

    return np.mean(train_loss)


def get_validate_loss(data_load, model, criterion, use_cuda=True):
    '''
    计算验证集损失
    :param data_load: 验证集
    :param model: 模型
    :param criterion: 评判标准
    :return:验证集平均损失
    '''

    # 切换模型为预测模型
    model.eval()
    # 定义用于保存验证集损失的列表
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, label) in enumerate(data_load):
            if use_cuda:  # 如果使用gpu训练 将tensor放到gpu
                input = input.cuda()
                label = label.cuda()

            y_hat = model(input)  # 获取预测值
            loss = criterion(y_hat, label)  # 计算损失
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(data_load, model, tta=3, use_cuda=True):
    '''
    模型预测
    :param data_load:
    :param model:
    :param tta:
    :return:
    '''

    '''
    TTA
    测试集数据扩增（Test Time Augmentation，简称TTA）也是常用的集成学习技巧，
    数据扩增不仅可以在训练时候用，而且可以同样在预测时候进行数据扩增，
    对同一个样本预测三次，然后对三次结果进行平均。
    '''
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(data_load):
                if use_cuda:
                    input = input.cuda()

                y_hat = model(input)
                if use_cuda:
                    output = y_hat.data.cpu().numpy()
                else:
                    output = y_hat.data.numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred  # array相加 tta次后概率相加

    return test_pred_tta


def get_validate_accuracy(val_loader, model):
    val_label = []
    val_predict = predict(val_loader, model).argmax(1)
    for i, (input, label) in enumerate(val_loader):
        val_label.append(label.data.numpy())
    val_char_acc = np.mean(val_predict == np.array(val_label).flatten())

    return val_char_acc


def save_model(model, optimizer, scheduler, is_last=False):
    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict()
    }

    if is_last:
        path = os.path.join('./save_weights', model.__class__.__name__, 'last_model.pth')
    else:
        path = os.path.join('./save_weights', model.__class__.__name__, 'model.pth')

    torch.save(save_files, path)


def load_weights_dict(model, optimizer=None, scheduler=None, is_last=False):
    if is_last:
        path = os.path.join('./save_weights', model.__class__.__name__, 'last_model.pth')
    else:
        path = os.path.join('./save_weights', model.__class__.__name__, 'model.pth')

    if path != '' and os.path.exists(path):
        weights_dict = torch.load(path)
        model.load_state_dict(weights_dict['model'])
        if optimizer is not None:
            optimizer.load_state_dict(weights_dict['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(weights_dict['lr_scheduler'])


def get_logger(path):
    # 1.显示创建
    # logging.basicConfig(filename='logg.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    # 2.定义logger,设定setLevel，FileHandler，setFormatter
    logger = logging.getLogger(path)  # 定义一次就可以，其他地方需要调用logger,只需要直接使用logger就行了
    if not logger.handlers:
        logger.setLevel(level=logging.INFO)  # 定义过滤级别
        filehandler = logging.FileHandler(path)  # Handler用于将日志记录发送至合适的目的地，如文件、终端等
        filehandler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        filehandler.setFormatter(formatter)

        console = logging.StreamHandler()  # 日志信息显示在终端terminal
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        logger.addHandler(filehandler)
        logger.addHandler(console)
    return logger


if __name__ == '__main__':
    for i in range(3):
        logger = get_logger('log.txt')
        logger.info('{}aaa{}'.format(1, 3))
        logger = get_logger('log2.txt')
        logger.info('{}aaa{}'.format(4, 2))
