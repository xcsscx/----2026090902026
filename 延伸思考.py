import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ---------- 基本参数 ----------
BATCH_SIZE = 64
NUM_CLASSES = 43          # GTSRB 共 43 类交通标志
EPOCHS = 10

# 测试集扰动幅度（只作用于测试集，训练集保持原样）
ROTATION_ANGLE = 10        # 小角度旋转（度）
BLUR_SIGMA = 1.0           # 高斯模糊强度
BRIGHTNESS_FACTOR = 1.3    # 亮度系数：>1 变亮，<1 变暗

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 1. 数据处理
# =========================================================
# GTSRB 原图尺寸不统一，先统一缩放到 32x32，再转张量并归一化
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                         std=[0.2672, 0.2564, 0.2629]),
])

trainset = torchvision.datasets.GTSRB(root="./data", split="train",
                                      download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.GTSRB(root="./data", split="test",
                                     download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集样本数: {len(trainset)}，测试集样本数: {len(testset)}，设备: {device}")


# ---- 构造不同扰动条件下的测试集 ----
def build_test_loader(perturb=None):
    """按指定扰动构建测试集加载器；perturb 为 None 时使用原始测试集。"""
    ops = [transforms.Resize((32, 32))]
    if perturb is not None:
        ops.append(perturb)
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                             std=[0.2672, 0.2564, 0.2629]),
    ]
    dataset = torchvision.datasets.GTSRB(
        root="./data", split="test", download=True,
        transform=transforms.Compose(ops))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


test_conditions = {
    "原始": testloader,
    "小角度旋转": build_test_loader(
        transforms.RandomRotation(degrees=(ROTATION_ANGLE, ROTATION_ANGLE))),
    "高斯模糊": build_test_loader(
        transforms.GaussianBlur(kernel_size=3, sigma=(BLUR_SIGMA, BLUR_SIGMA))),
    "亮度调整": build_test_loader(
        transforms.ColorJitter(brightness=(BRIGHTNESS_FACTOR, BRIGHTNESS_FACTOR))),
}


# =========================================================
# 2. 构建 CNN 模型
# =========================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 个卷积块：卷积 -> BatchNorm -> ReLU -> 最大池化
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),                   # 批归一化：稳定分布、加快收敛
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 8 -> 4
        )
        # 全连接分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),          # 随机丢弃 50% 的隐藏层神经元，抑制过拟合
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN().to(device)

# =========================================================
# 3. 损失函数与优化器
# =========================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# =========================================================
# 4. 训练与评估
# =========================================================
def evaluate(model, loader, criterion):
    """返回测试集上的平均损失和准确率。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, EPOCHS + 1):
    # ---- 训练一个 epoch ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- 在测试集上评估 ----
    test_loss, test_acc = evaluate(model, testloader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch [{epoch}/{EPOCHS}] | "
          f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
          f"test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")


# =========================================================
# 5. 鲁棒性测试：原始 vs 旋转 / 模糊 / 亮度
# =========================================================
print("\n===== 不同测试条件下的准确率 =====")
base_acc = None
for name, loader in test_conditions.items():
    loss, acc = evaluate(model, loader, criterion)
    if base_acc is None:
        base_acc = acc
    print(f"{name:<6} | loss: {loss:.4f}, acc: {acc:.4f}, "
          f"相对原始下降: {base_acc - acc:+.4f}")
