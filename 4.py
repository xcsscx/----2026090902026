import torch
import torch.nn as nn

# 1. 基本参数
batch_size = 2
seq_len = 4
embed_dim = 8
num_heads = 2
num_layers = 2

# 2. 构造输入
x = torch.randn(
    batch_size,
    seq_len,
    embed_dim
)

# 3. 构建 Transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    batch_first=True
)

encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=num_layers
)

# 4. 前向传播
out = encoder(x)

# 5. 查看结果
print("Input shape:", x.shape)
print("Output shape:", out.shape)
print("输出与输入 shape 是否一致:", out.shape == x.shape)

# 6. 尝试修改参数：num_heads 2 -> 4，再运行一次
num_heads = 4

encoder_layer_new = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    batch_first=True
)

encoder_new = nn.TransformerEncoder(
    encoder_layer_new,
    num_layers=num_layers
)

out_new = encoder_new(x)

print("\n修改 num_heads 后：")
print("Output shape:", out_new.shape)
