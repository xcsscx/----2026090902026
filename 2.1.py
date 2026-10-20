import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================= 1. 数据生成与预处理 =================
# 按题目要求生成数据
X, y = make_friedman1(
    n_samples=1500,
    n_features=10,
    noise=1.0,
    random_state=42
)

# 划分训练集 / 测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 转成 PyTorch 张量（float32）；y 用 view(-1, 1) 使其形状与模型输出 (N, 1) 对齐
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ================= 2. 搭建线性回归模型（不含隐藏层） =================
class LinearRegressionModel(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)

# ================= 3. 模型训练 =================
criterion = nn.MSELoss()                                  # 回归任务常用均方误差
optimizer = optim.Adam(model.parameters(), lr=0.05)       # 优化器

EPOCHS = 1000
loss_history = []

model.train()
for epoch in range(EPOCHS):
    y_pred = model(X_train_t)          # 前向传播
    loss = criterion(y_pred, y_train_t)

    optimizer.zero_grad()              # 梯度清零
    loss.backward()                    # 反向传播
    optimizer.step()                   # 更新参数

    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Train MSE Loss: {loss.item():.4f}")

# 训练完成后打印学到的权重与偏置
w = model.linear.weight.detach().numpy().flatten()
b = model.linear.bias.item()
print("\n学到的权重 w =", np.round(w, 4))
print(f"学到的偏置 b = {b:.4f}")

# ================= 4. 模型测试与评价 =================
model.eval()
with torch.no_grad():
    y_pred_np = model(X_test_t).numpy().flatten()
y_test_np = y_test_t.numpy().flatten()

mse = mean_squared_error(y_test_np, y_pred_np)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_np, y_pred_np)
r2 = r2_score(y_test_np, y_pred_np)

print("\n================= 测试集评价指标 =================")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"R^2      : {r2:.4f}")

