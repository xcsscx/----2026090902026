# 任务一：完成一次大模型 API 调用

## 运行方法

1. 安装依赖（在项目根目录的 `.venv` 中执行）：

```powershell
pip install openai python-dotenv
```

2. 在项目根目录创建 `.env`，填入自己的密钥：

```dotenv
API_KEY=你的_API_Key
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat
```

3. 运行程序：

```powershell
.\.venv\Scripts\python.exe 5.1.py
```

4. 按提示输入问题并回车即可获得回答，输入 `q` 退出。

## 所用模型

`deepseek-chat`

## 所用平台

DeepSeek 开放平台（`https://api.deepseek.com`），通过 OpenAI 兼容接口调用。
