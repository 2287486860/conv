import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import f1_score, recall_score, precision_score
import wandb
import torchvision
from tqdm import tqdm
# from models import AlexNet,GoogLeNet,VGG,ResNet,ShuffleNet,DenseNet,Xception,MobileNet,vgg0
from Xception import self

import os

model = self.model
num_classes=38
# 定义训练和测试函数
import argparse
os.environ['WANDB_API_KEY'] = 'ce488c1a2ad2ca73d47c1c5f555dea59e14fadcc'
wandb.init(project='flower')
# 登录WandB
wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
args = parser.parse_args()
def train(model, dataloader, criterion, optimizer, device, train_step_freq=100):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    step = 0
    for inputs, labels in tqdm(dataloader, desc='Train', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        step += 1
        if step % train_step_freq == 0:
            epoch_loss = running_loss / (train_step_freq * dataloader.batch_size)
            epoch_acc = running_corrects.double() / (train_step_freq * dataloader.batch_size)
            print(f'Train Step {step}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}')
            running_loss = 0.0
            running_corrects = 0
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def test(model, dataloader, criterion, device, test_step_freq=50):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    y_true = []
    y_pred = []
    step = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Test', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            y_true += labels.cpu().numpy().tolist()
            y_pred += preds.cpu().numpy().tolist()
            step += 1
            if step % test_step_freq == 0:
                epoch_loss = running_loss / (test_step_freq * dataloader.batch_size)
                epoch_acc = running_corrects.double() / (test_step_freq * dataloader.batch_size)
                epoch_f1 = f1_score(y_true, y_pred, average='macro')
                epoch_recall = recall_score(y_true, y_pred, average='macro')
                epoch_precision = precision_score(y_true, y_pred, average='macro')
                print(f'Test Step {step}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, f1={epoch_f1:.4f}, recall={epoch_recall:.4f}, precision={epoch_precision:.4f}')
                running_loss = 0.0
                running_corrects = 0
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_f1 = f1_score(y_true, y_pred, average='macro')
    epoch_recall = recall_score(y_true, y_pred, average='macro')
    epoch_precision = precision_score(y_true, y_pred, average='macro')
    return epoch_loss, epoch_acc, epoch_f1, epoch_recall, epoch_precision

# 加载数据集
train_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset =torchvision.datasets.ImageFolder(root='./data/train', transform=train_transform)
test_dataset =torchvision.datasets.ImageFolder(root='./data/test', transform=test_transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 定义模型、损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr =args.lr, momentum=0.9, weight_decay=5e-4)
# 设置学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 将模型和优化器添加到WandB

# wandb.watch(model)
# wandb.watch(optimizer)

# 训练和测试模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
best_acc = 0.0
for epoch in range(30):
    train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device, train_step_freq=100)

    test_loss, test_acc, test_f1, test_recall, test_precision = test(model, test_dataloader, criterion, device,
                                                                     test_step_freq=50)
    if test_acc > best_acc:
        best_acc = test_acc
    # 记录指标到WandB

    wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
               'test_acc': test_acc, 'test_f1': test_f1, 'test_recall': test_recall, 'test_precision': test_precision,
               'best_acc': best_acc})
    # 更新学习率
    scheduler.step()

# 关闭WandB
wandb.finish()