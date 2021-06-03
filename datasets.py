import collections
import math
import os
import shutil

import torch
import torchvision

data_dir = './data/'  # 数据文件夹
valid_ratio = 0.2  # 验证集比率
batch_size = 64  # 批量大小


def read_csv_labels(fname):
    """
    读取文件
    :param fname:
    :return:返回文件名和目标值
    """
    with open(fname, 'r') as f:
        # 跳过文件头行 (列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


def copyfile(filename, target_dir):
    """
    复制文件
    :param filename:文件的绝对路径
    :param target_dir: 目标文件夹
    :return:
    """
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    '''
    处理训练集
    :param data_dir:数据文件夹
    :param labels:文件名:目标值 字典
    :param valid_ratio:验证集比率
    :return:验证集中每个类别的示例数
    '''
    # 训练数据集中示例最少的类别中的示例数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的示例数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file, label in labels.items():
        # 将所有训练集的数据放到新目录下
        fname = os.path.join(data_dir, train_file)
        copyfile(
            fname,
            os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        # 保存验证集数据
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:  # 保存训练集数据
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label


def reorg_test(data_dir):
    '''
    处理测试集数据
    :param data_dir:数据所在文件夹
    :return:
    '''
    # 读取测试集csv 拷贝测试集文件
    with open(os.path.join(data_dir, 'test.csv'), 'r') as f:
        for test_file in f.readlines()[1:]:
            fname = os.path.join(data_dir, test_file.strip())
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))


def reorg_data(data_dir, valid_ratio):
    '''
    处理所有数据
    :param data_dir:数据文件夹
    :param valid_ratio:验证集比率
    :return:
    '''
    labels = read_csv_labels(os.path.join(data_dir, 'train.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


# 训练集图片转换
transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(299),#INCEPTIONV3
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64到1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    # torchvision.transforms.CenterCrop((180, 180)),
    # torchvision.transforms.RandomResizedCrop(180, scale=(0.5, 1.0),
    #                                          ratio=(1.0, 1.0)),
    # torchvision.transforms.ColorJitter(0.3, 0.3, 0.2),
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    # torchvision.transforms.Normalize([0.73628336, 0.7584505, 0.7314892], [0.17110135, 0.16181387, 0.20289394])
])
# 测试集图片转换
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(299),#INCEPTIONV3
    # torchvision.transforms.CenterCrop((180, 180)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize([0.7384071, 0.75999546, 0.7332944], [0.17157324, 0.1621383, 0.20280267])
])


def get_image_list(img_dir, isclasses=False):
    """获取图像的名称列表
    args: img_dir:存放图片的目录
          isclasses:图片是否按类别存放标志
    return: 图片文件名称列表
    """
    img_list = []
    # 路径下图像是否按类别分类存放
    if isclasses:
        img_file = os.listdir(img_dir)
        for class_name in img_file:
            if not os.path.isfile(os.path.join(img_dir, class_name)):
                class_img_list = os.listdir(os.path.join(img_dir, class_name))
                img_list.extend([os.path.join(class_name, fname) for fname in class_img_list])
    else:
        img_list = os.listdir(img_dir)
    # print(img_list)
    print('image numbers: {}'.format(len(img_list)))
    return img_list


def getStat(train_data):
    '''
    计算训练数据的平均值和方差
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)均值和方差用于数据增强
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


def get_mean_std():
    '''
    获取训练集测试集均值方差
    :return:
    '''
    train_ds, train_valid_ds = [
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train_valid_test', folder),
            transform=transform_train) for folder in ['train', 'train_valid']]

    valid_ds = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', 'valid'),
        transform=transform_test)

    test_ds = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', 'test'),
        transform=transform_test)

    print("train_ds", getStat(train_valid_ds))
    # 18353 train_ds([0.73628336, 0.7584505, 0.7314892], [0.17110135, 0.16181387, 0.20289394])
    print("test_ds", getStat(test_ds))
    # 8800 test_ds([0.7384071, 0.75999546, 0.7332944], [0.17157324, 0.1621383, 0.20280267])


class Singleton:
    # __new__() 方法是在类准备将自身实例化时调用。
    # __new__() 方法始终都是类的静态方法，即使没有被加上静态方法装饰器。
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class ClassLeavesDataSet(Singleton):
    def __init__(self, batch_size=16):
        self.train_ds, self.train_valid_ds = [
            torchvision.datasets.ImageFolder(
                os.path.join(data_dir, 'train_valid_test', folder),
                transform=transform_train) for folder in ['train', 'train_valid']]

        self.valid_ds, self.test_ds = [torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train_valid_test', folder),
            transform=transform_test) for folder in ['valid', 'test']]

        self.train_iter, self.train_valid_iter = [
            torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                        drop_last=True)
            for dataset in (self.train_ds, self.train_valid_ds)]

        self.valid_iter = torch.utils.data.DataLoader(self.valid_ds, batch_size, shuffle=False,
                                                      drop_last=True)

        self.test_iter = torch.utils.data.DataLoader(self.test_ds, batch_size, shuffle=False,
                                                     drop_last=False)


if __name__ == '__main__':
    # 测试数据加载
    clds = ClassLeavesDataSet()
    unloader = torchvision.transforms.ToPILImage()
    for i, (features, labels) in enumerate(clds.train_iter):
        for f in features[:10]:
            img_ = f.cpu().clone()
            img_ = img_.squeeze(0)
            img_ = unloader(img_)
            img_.show()
        break
    # for i, (features, labels) in enumerate(valid_iter):
    #     print(i, (features, labels))
    #     break
    # for i, (features, labels) in enumerate(train_valid_iter):
    #     print(i, (features, labels))
    #     break

    # test_iter = get_test_iter()
    # for X, _ in test_iter:
    #     print(X)
    #     break
    # 将数据通过类别划分训练集验证集测试集
    # reorg_data(data_dir, valid_ratio)
    # get_mean_std()
