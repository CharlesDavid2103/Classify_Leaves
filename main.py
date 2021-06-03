from train import train_and_validate
from test_submit import test
import model


def main():
    for train_model in [
                        # model.INITResNet18(),
                        # model.INITResNet34(),
                        # model.INITResNet50(),
                        # model.INITResNet101(),
                        # model.INITResNet152(),
                        # model.DENSENET121(),
                        # model.DENSENET161(),
                        # model.DENSENET169(),
                        # model.DENSENET201(),
                        model.INCEPTIONV3(),
                        ]:
        train_and_validate(train_model)
        test(train_model)


if __name__ == '__main__':
    main()
