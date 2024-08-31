import matplotlib.pyplot as plt
import os
import torch
import time
import pandas as pd
def draw_acc():
    # 读取保存的csv文件，获取训练集、测试集的损失值和准确率
    filename = r'./mob2.csv'
    train_data = pd.read_csv(filename)
    # print(train_data.head())

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



    draw_acc()
