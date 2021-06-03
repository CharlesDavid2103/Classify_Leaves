import os
import torch
from torch import nn
from datasets import ClassLeavesDataSet
from utils import train, get_validate_loss, get_validate_accuracy, load_weights_dict, save_model, get_logger

clds = ClassLeavesDataSet()


def train_and_validate(model, use_cuda=True):
    '''
    模型训练、预测
    :param use_cuda:是否使用gpu
    '''
    # 日志
    logger = get_logger(os.path.join('./save_weights', model.__class__.__name__, 'log.txt'))
    logger.info('strat train')
    # 加载数据集
    train_iter, valid_iter, train_valid_iter = clds.train_iter, clds.valid_iter, clds.train_valid_iter

    # 定义模型和学习率  学习率衰减的迭代次数、衰减率
    lr, lr_period, lr_decay = 1e-3, 10, 0.1
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)

    bast_loss_path = os.path.join('./save_weights', model.__class__.__name__, 'best_loss')

    if use_cuda:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    if os.path.exists(bast_loss_path):
        bast_param = torch.load(bast_loss_path)
        best_loss = float(bast_param['best_loss'])
        best_val_loss = float(bast_param['best_val_loss'])
        load_weights_dict(model, optimizer, scheduler)
    else:
        best_loss = 1000.0
        best_val_loss = 1000.0

    count = 0
    for epoch in range(100):
        train_loss = train(train_valid_iter, model, criterion, optimizer)
        val_loss = get_validate_loss(valid_iter, model, criterion)
        scheduler.step()
        val_acc = get_validate_accuracy(valid_iter, model)
        logger.info('Epoch: {0}, Train loss: {1} \t Val loss: {2} \t Val Acc: {3}'.format(epoch, train_loss, val_loss,
                                                                                          val_acc))

        if train_loss < best_loss and val_loss < best_val_loss:
            best_loss = train_loss
            best_val_loss = val_loss

            save_file = {
                'best_loss': best_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(save_file, bast_loss_path)
            save_model(model, optimizer, scheduler)
            count = 0
        else:
            count += 1

        if count >= 5:
            logger.info('over')
            logger.info('Train loss: {0} \t Val loss: {1} '.format(best_loss, best_val_loss))
            save_model(model, optimizer, scheduler, is_last=True)
            break

        if (epoch + 1) % 10 == 0:
            save_model(model, optimizer, scheduler, is_last=True)

    save_model(model, optimizer, scheduler, is_last=True)
    logger.info('end train')


if __name__ == '__main__':
    import model
    train_and_validate(model.DENSENET121())
