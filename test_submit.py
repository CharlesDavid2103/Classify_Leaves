import pandas as pd
import torch
import os
from datasets import ClassLeavesDataSet

from utils import load_weights_dict, predict, get_logger

torch.backends.cudnn.benchmark = True

clds = ClassLeavesDataSet()


def test(model, use_cuda=True):
    logger = get_logger(os.path.join('./save_weights', model.__class__.__name__, 'log.txt'))
    logger.info('strat test')

    train_ds, valid_ds, train_valid_ds = clds.train_ds, clds.valid_ds, clds.train_valid_ds
    load_weights_dict(model)

    if use_cuda:
        model = model.cuda()

    model.eval()
    test_iter = clds.test_iter
    preds = predict(test_iter, model, 10).argmax(1)

    df_submit = pd.read_csv('./data/sample_submission.csv')
    df_submit['label'] = preds
    df_submit['label'] = df_submit['label'].apply(lambda x: train_valid_ds.classes[x])
    df_submit.to_csv(os.path.join('./save_weights', model.__class__.__name__, 'submit.csv'), index=None)
    logger.info('end test')


if __name__ == '__main__':
    test()
