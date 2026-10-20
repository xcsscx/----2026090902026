import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 固定随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================= 1. 数据生成与预处理 =================
X, y = make_friedman1(
    n_samples=1500,
    n_features=10,
    noise=1.0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ================= 2. 全连接神经网络定义 =================
class MLP(nn.Module):
    """全连接神经网络：2 个隐藏层 (64, 32) + ReLU 激活，输出 1 维"""

    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters())


mlp_model = MLP(X_train.shape[1])


# ================= 3. 训练 =================
def train_model(model, epochs=1000, lr=0.01):
    """通用的训练函数：全批量梯度下降，返回每个 epoch 的损失"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()
    for epoch in range(epochs):
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


EPOCHS = 1000
loss_mlp = train_model(mlp_model, EPOCHS, lr=0.01)


# ================= 4. 测试与评价 =================
def evaluate(model):
    """在训练集与测试集上评估，返回指标与预测值"""
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).numpy().flatten()
        pred_train = model(X_train_t).numpy().flatten()
    yt = y_test_t.numpy().flatten()
    ytr = y_train_t.numpy().flatten()
    return {
        'pred_test': pred_test,
        'mse': mean_squared_error(yt, pred_test),
        'rmse': np.sqrt(mean_squared_error(yt, pred_test)),
        'mae': mean_absolute_error(yt, pred_test),
        'r2': r2_score(yt, pred_test),
        'train_r2': r2_score(ytr, pred_train),
    }


res_mlp = evaluate(mlp_model)

print("\n========== 最终测试结果 ==========")
print(f"MLP: MSE={res_mlp['mse']:.4f}, RMSE={res_mlp['rmse']:.4f}, "
      f"MAE={res_mlp['mae']:.4f}, 测试R2={res_mlp['r2']:.4f}, "
      f"训练R2={res_mlp['train_r2']:.4f}")
