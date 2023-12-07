import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import MobileNetV2
import torchvision.models.mobilenet
'''
这段代码是一个使用PyTorch进行迁移学习的例子，主要完成以下几个任务：

1. 数据预处理和加载：
    使用torchvision中的transforms对训练集和验证集进行预处理，包括裁剪、翻转和标准化等操作。
    使用ImageFolder类加载数据集，并创建对应的数据加载器DataLoader。

2. 构建模型：
    使用了在model.py文件中定义的MobileNetV2模型。
    加载预训练的MobileNetV2模型权重，并去除最后的分类器部分。
    冻结模型的特征提取部分的参数，使其不参与训练。
    将模型移动到GPU上（如果可用）。

3. 定义损失函数和优化器：
    使用交叉熵损失作为损失函数。
    选择Adam优化器，并传入模型的可训练参数和学习率。

4. 训练和验证：
    循环遍历多个epoch，在每个epoch中进行训练和验证。
    训练过程中，通过数据加载器逐批次获取数据，计算模型输出、损失和梯度，并更新模型参数。
    验证过程中，关闭梯度计算，计算模型在验证集上的准确率，并保存在验证集上表现最好的模型参数。

5. 最终保存模型参数：
    在整个训练过程结束后，保存在验证集上表现最好的模型参数到文件road.pth中。
    这段代码的主要目的是利用迁移学习，使用预训练的MobileNetV2模型，在自定义的道路数据集上进行微调，以达到对道路场景进行分类的目的。
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
}

data_root = os.path.abspath(os.path.join(os.getcwd(), "."))  # get data root path
image_path = data_root + "/road_data/"  # flower data set path

train_dataset = datasets.ImageFolder(
    root=image_path + "train",
    transform=data_transform["train"]
)
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

validate_dataset = datasets.ImageFolder(
    root=image_path + "val",
    transform=data_transform["val"]
)

val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(
    validate_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

net = MobileNetV2(num_classes=5)

# 常见卷积神经网络（CNN），主要由几个 卷积层Conv2D 和 池化层MaxPooling2D 层组成。卷积层与池化层的叠加实现对输入数据的特征提取，最后连接全连接层实现分类。
# 特征提取——卷积层与池化层
# 实现分类——全连接层
# 这里用到“迁移学习”的思想，使用“预训练模型”作为特征提取；实现分类的全连接层有我们自己搭建。


# load pretrain weights
# 加载预训练的MobileNetV2模型权重
model_weight_path = "transfer_model/mobilenet_v2.pth"
pre_weights = torch.load(model_weight_path)

# delete classifier weights
pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

"""
删除预训练模型的分类器部分是因为在进行迁移学习时，一般会将预训练模型的特征提取部分保留下来，而不使用其原有的分类器部分。这是因为：

1. 适应新任务：预训练模型的特征提取部分已经包含了对图像特征的抽取能力，可以帮助我们适应新的数据集和任务。因此，通常会保留这部分权重，而不需要重新训练。

2. 避免过拟合：预训练模型的分类器部分通常是为了原始数据集的特定分类任务而设计的，如果直接在新数据集上使用，可能会导致过拟合。因此，我们会删除原有的分类器，通过微调或者添加新的分类器来适应新的任务。

3. 灵活性：删除原有的分类器后，可以根据新任务的需求自由地添加新的分类器结构，比如改变输出类别数、调整全连接层的结构等，从而更好地适应新任务。

因此，在这段代码中，删除预训练模型的分类器部分是为了在自定义的道路数据集上进行微调时，能够更好地适应新的分类任务。
"""

'''
# freeze features weights
for param in net.features.parameters():
    param.requires_grad = False
'''
net.to(device)

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# print(net)
best_acc = 0.0
save_path = 'road.pth'
for epoch in range(20):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')
