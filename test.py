######################################
#                                    #
#     Train CIFAR10 with PyTorch     #
#                                    #
######################################


from torchinfo import summary
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse
# import resmlp
# import att
from torch.utils.tensorboard import SummaryWriter
from  models import VGG,GoogLeNet,AlexNet,Xception
import torch.nn as nn
import wandb
wandb.init(project='flower',name='test')
# Create a TensorBoard summary writer
# writer = SummaryWriter('runs/experiment_1')
model = AlexNet.model
# 创建ArgumentParser对象，其description参数为脚本的简单描述。
parser = argparse.ArgumentParser(description='Train CIFAR10 with PyTorch')

# 添加命令行参数：
# --lr : 参数名
# help : 帮助信息，指定了该参数的作用及说明
# default=0.1 : 指定该参数的默认数值为0.1
# type=float : 指定了这个参数的输入类型是float类型
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

# 添加更多命令行参数，其中 '-r' 为 '--resume' 的简写形式，
# 当执行脚本时带有 '-r' 参数时，args.resume == True，否则为False。
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# 解析命令行参数，并返回一个命名空间。该命名空间属性名称与参数名称保持一致。
args = parser.parse_args()

# 定义 Cifar10 中的类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 引入Transform类和Compose类，用于对数据进行预处理
from torchvision import transforms

# 定义batch_size为128，表示每次读入训练的数据量为128个样本。
batch_size = 128

# 定义 transform_train 和 transform_test 对象，分别用于对训练集和测试集进行图像变换
# Compose类将多个转换函数组合起来应用到数据上，以RandomCrop、RandomHorizontalFlip和ToTensor等函数依次作用于每幅图像上，
# 可同时进行多种变换，并在训练时利用多样性提高模型的鲁棒性
transform_train = transforms.Compose([
    # transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 测试数据集不需进行任何增强，因此仅需使用 ToTensor 和 Normalize 等固定变换即可
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载 CIFAR10 数据集，训练集和测试集分别存储在train_set和test_set中
train_set = torchvision.datasets.ImageFolder(
    root='./data/train',  transform=transform_train)

# 使用 DataLoader 将训练数据集划分为多个大小为128的batch，把各个batch送入模型训练
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.ImageFolder(
    root='./data/test',  transform=transform_test)

# 构造测试集 DataLoader，shuffle=False，表示不打乱测试集样本顺序。
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)
# 打印检查深度学习 CNN 模型
print('------ Check CNN Model ------')

# 选择使用的深度学习 CNN 模型，这里提供了多个预定义模型可选：
# AlexNet、VGG11/13/16/19、GoogLeNet、ResNet18/34/50/101/152、DenseNet121/161/169/201/264、
# SE_ResNet50、CBAM_ResNet50、ECA_ResNet50、squeezenet1_0/1_1、MobileNet、shufflenet_g1/g2/g3/g4/g8、Xception
# 可以根据实际需求在这里进行选择，例如：net = ResNet18()
# 未被注释掉的模型作为参考
best_acc = 0
start_epoch = 0
end_epoch = 3


# 设置 GPU/CPU 运行环境
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判断 CUDA 是否可用，如果可用，则使用 GPU 运行，否则使用 CPU
print('cuda is available : ', torch.cuda.is_available())  # 输出 CUDA 是否可用的结果
model = model.to(device)  # 将模型加载到对应设备上，加快运算速度

# 如果 args.resume 为 True，则恢复上一次模型训练状态
if args.resume:
    print('------ Loading checkpoint ------')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'  # 如果目录不存在，则输出错误信息
    checkpoint = torch.load('./checkpoint/ckpt.pth')  # 加载之前训练好的模型参数，包括轮数和准确率等指标
    model.load_state_dict(checkpoint['model'])  # 加载之前训练好的模型的参数列表到当前模型中
    best_acc = checkpoint['acc']  # 记录该模型在验证集上达到的最佳准确率
    start_epoch = checkpoint['epoch']  # 加载之前训练中断的轮数，从该轮开始重新开始训练
    end_epoch += start_epoch  # 继承从上次的轮数累加到增加的训练 epoch 上

# 设置损失函数为交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 设置优化器为批量随机梯度下降算法，学习率、动量和权重衰减分别设为 args.lr、0.9 和 5e-4
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# 设置 Multiple LR Scheduling 策略，每经过20个epoch，学习率乘以gamma（gamma 默认为0.1）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)


# 定义对模型进行摘要的函数，用于输出模型结构信息
def model_summary():
    print('------ Model Summary ------')
    # 随机输入一个数据，可以获取到模型中每一层的输出大小
    y = model(torch.randn(1, 3, 32, 32).to(device))
    # 打印出最终的神经元数量
    print(y.size())
    # 使用summary库来输出模型的详细结构信息，包括所有层的名称，类型，输入和输出大小等
    summary(model, (1, 3, 32, 32), depth=5)


##########################

# Training
##########################

# 定义训练函数，输入指定轮次(epoch)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # 将模型切换到"train"模式，即开启训练状态，dropout、batchnorm层等会开始工作
    model.train()
    # 初始化训练损失、预测正确数和总数
    train_loss = 0
    correct = 0
    total = 0

    # 枚举训练集数据的批次和相应的标签(targets)，计算训练损失、准确率和预测结果并更新模型参数
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # 清空上个批次的梯度信息
        outputs = model(inputs)  # 推送模型数据到CNN网络中去进行前向传播
        loss = criterion(outputs, targets) # 使用CrossEntropy Loss函数计算损失值
        loss.backward()  # 反向传播，计算梯度并更新参数
        optimizer.step()  # 更新模型的权重参数

        train_loss += loss.item()  # 累加每个batch的损失函数值
        _, predicted = outputs.max(1)  # 获取最高分数预测的类别
        total += targets.size(0)  # 累加当前批次的标签数量
        correct += predicted.eq(targets).sum().item()  # 判断预测是否正确，累加正确的个数

        # writer.add_scalar('Train Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('Train Accuracy', correct / total, epoch * len(train_loader) + batch_idx)
        # 每50个batch输出一次当前轮次的平均训练损失和准确率
        if batch_idx % 50 == 0:
            print('\tLoss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    # wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc})

    print('\n', time.asctime(time.localtime(time.time())))
    print(' Epoch: %d | Train_loss: %.3f | Train_acc: %.3f%% \n' % (epoch, train_loss, train_acc))
    wandb.log({"Train Loss": train_loss, "Train Acc uracy": train_acc})
    return train_loss, train_acc


def test(epoch):
    # 全局变量，记录最好的准确率
    global best_acc
    # 切换到“评估”模式
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    tp = 0  # true positive
    tn = 0  # true negative
    fp = 0  # false positive
    fn = 0  # false negative
    e=1e-8
    with torch.no_grad():
        # tqdm 用于展示进度条
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 计算tp、tn、fp、fn
            tp += ((predicted == targets) & (targets == 1)).sum().item()
            tn += ((predicted == targets) & (targets == 0)).sum().item()
            fp += ((predicted != targets) & (targets == 0)).sum().item()
            fn += ((predicted != targets) & (targets == 1)).sum().item()

            # writer.add_scalar('Test Loss', loss.item(), epoch * len(test_loader) + batch_idx)
            # writer.add_scalar('Test Accuracy', correct / total, epoch * len(test_loader) + batch_idx)
            if batch_idx % 50 == 0:
                # 每50个batch打印一次损失和准确率
                print('\tLoss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # 测试集损失值和准确率
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total

        # 计算recall、F1、predict等指标
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = 2 * precision * recall / (precision + recall)
        # predict = (tp + tn) / (tp + tn + fp + fn)

        # 打印当前时间、当前轮次测试结果
        print('\n', time.asctime(time.localtime(time.time())))
        print(' Epoch: %d | Test_loss: %.3f | Test_acc: %.3f%% | Recall: %.3f | F1 Score: %.3f | Precision: %.3f \n' %
              (epoch, test_loss, test_acc, recall, f1_score, precision))
        wandb.log({"Test Loss": test_loss, "Test Accuracy": test_acc,"Test recall":recall,"Test f1":f1_score,"Test precision":precision,"Test best":best_acc})
    if test_acc > best_acc:
        print('------ Saving model------')
        state = {
            'model': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/model_%d_%.3f.pth' % (epoch, best_acc))
        best_acc = test_acc

    # return test_loss, test_acc, recall, f1_score, precision,best_acc
    return test_loss, test_acc

# 此函数用于保存训练和测试的损失值和准确率到csv文件中
def save_csv(epoch, save_train_loss, save_train_acc, save_test_loss, save_test_acc):
    # 获取当前时间
    time = '%s' % datetime.now()
    # step 记录轮次
    step = 'Step[%d]' % epoch
    train_loss = '%f' % save_train_loss
    train_acc = '%g' % save_train_acc
    test_loss = '%f' % save_test_loss
    test_acc = '%g' % save_test_acc

    print('------ Saving csv ------')
    list = [time, step, train_loss, train_acc, test_loss, test_acc]

    # 将数据转化为DataFrame格式，并保存到csv文件中
    data = pd.DataFrame([list])
    data.to_csv('./train_acc.csv', mode='a', header=False, index=False)


# 此函数用于绘制训练集、测试集的损失值和准确率图像
def draw_acc():
    # 读取保存的csv文件，获取训练集、测试集的损失值和准确率
    filename = r'./train_acc.csv'
    train_data = pd.read_csv(filename)
    print(train_data.head())

    length = len(train_data['step'])
    Epoch = list(range(1, length + 1))

    train_loss = train_data['train loss']
    train_accuracy = train_data['train accuracy']
    test_loss = train_data['test loss']
    test_accuracy = train_data['test accuracy']

    # 使用Matplotlib库将训练集、测试集的损失值和准确率在同一张图中绘制出来。
    plt.plot(Epoch, train_loss, 'g-.', label='train loss')
    plt.plot(Epoch, train_accuracy, 'r-', label='train accuracy')
    plt.plot(Epoch, test_loss, 'b-.', label='test loss')
    plt.plot(Epoch, test_accuracy, 'm-', label='test accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss & Accuracy')
    plt.yticks([j for j in range(0, 101, 10)])
    plt.title('Epoch -- Loss & Accuracy')
    plt.legend(loc='center right', fontsize=8, frameon=False)
    plt.show()


if __name__ == '__main__':

    # model_summary()
    if not os.path.exists('../GCN/train_acc.csv'):
        df = pd.DataFrame(columns=['time', 'step', 'train loss', 'train accuracy', 'test loss', 'test accuracy'])
        df.to_csv('./train_acc.csv', index=False)
        print('make csv successful !')
    else:
        print('csv is exist !')

    for epoch in range(start_epoch, end_epoch):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        scheduler.step()

        # save_csv(epoch, train_loss, train_acc, test_loss, test_acc)
    # writer.close()
    # draw_acc()
