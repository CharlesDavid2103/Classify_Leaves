import torchvision
import os
from torch import nn


# Epoch: 19, Train loss: 0.032505877573135215 	 Val loss: 0.17293008757247166 	 Val Acc: 0.9545454545454546
class INITResNet18(nn.Sequential):
    def __init__(self):
        super(INITResNet18, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练resnet18
        self.features = torchvision.models.resnet18(pretrained=True)

        self.features.fc = nn.Linear(512, num_classes)


class INITResNet34(nn.Sequential):
    def __init__(self):
        super(INITResNet34, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet34

        self.features = torchvision.models.resnet34(pretrained=True)
        # 设置输出维度
        self.features.fc = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout(0.5)


class INITResNet50(nn.Sequential):
    def __init__(self):
        super(INITResNet50, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet50

        self.features = torchvision.models.resnet50(pretrained=True)
        # 设置输出维度
        self.features.fc = nn.Linear(2048, num_classes)
        # self.dropout = nn.Dropout(0.5)


class INITResNet101(nn.Sequential):
    def __init__(self):
        super(INITResNet101, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet101

        self.features = torchvision.models.resnet101(pretrained=True)
        # 设置输出维度
        self.features.fc = nn.Linear(2048, num_classes)
        # self.dropout = nn.Dropout(0.5)


# Epoch: 14, Train loss: 0.027745427699924205 	 Val loss: 0.05467102479729378 	 Val Acc: 0.9806818181818182
class INITResNet152(nn.Sequential):
    def __init__(self):
        super(INITResNet152, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet269

        self.features = torchvision.models.resnet152(pretrained=True)
        # 设置输出维度
        self.features.fc = nn.Linear(2048, num_classes)


class DENSENET121(nn.Sequential):
    def __init__(self):
        super(DENSENET121, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet269

        self.features = torchvision.models.densenet121(pretrained=True)
        # 设置输出维度
        self.features.classifier = nn.Linear(1024, num_classes)


class DENSENET161(nn.Sequential):
    def __init__(self):
        super(DENSENET161, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet269

        self.features = torchvision.models.densenet161(pretrained=True)
        # 设置输出维度
        self.features.classifier = nn.Linear(2208, num_classes)


class DENSENET169(nn.Sequential):
    def __init__(self):
        super(DENSENET169, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet269

        self.features = torchvision.models.densenet169(pretrained=True)
        # 设置输出维度
        self.features.classifier = nn.Linear(1664, num_classes)


class DENSENET201(nn.Sequential):
    def __init__(self):
        super(DENSENET201, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet269

        self.features = torchvision.models.densenet201(pretrained=True)
        # 设置输出维度
        self.features.classifier = nn.Linear(1920, num_classes)


class INCEPTIONV3(nn.Sequential):
    def __init__(self):
        super(INCEPTIONV3, self).__init__()

        path = os.path.join('./save_weights', self.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # 预测类别数
        num_classes = 176
        # 定义一个moduel
        # 使用预训练INITResNet269

        self.features = torchvision.models.inception_v3(pretrained=True)
        # 设置输出维度
        self.features.fc = nn.Linear(2048, num_classes)



if __name__ == '__main__':
    # print([e for e in dir(torchvision.models) if not e.startswith('_')])
    model = INCEPTIONV3()
    print(model.__class__.__name__)
    print(model)
