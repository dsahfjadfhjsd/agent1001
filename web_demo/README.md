# Agent1001 Web Demo - 依赖安装指南

## 快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate agent1001
```

### 2. 自动安装（推荐）
```bash
# Windows用户
install_dependencies.bat

# 或手动安装
pip install -r requirements.txt
```

### 3. 手动安装步骤
```bash
# 1. 安装PyTorch (GPU版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. 安装其他依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "import torch, flask, flask_socketio; print('安装成功')"
```

## 核心依赖说明

### Web框架
- **Flask**: Web应用框架
- **Flask-SocketIO**: 实时WebSocket通信
- **eventlet**: 异步事件处理

### AI/ML核心
- **torch**: PyTorch深度学习框架
- **transformers**: Hugging Face模型库
- **sentence-transformers**: 句子嵌入模型
- **langchain**: LLM应用框架
- **openai**: OpenAI API客户端

### 数据处理
- **pandas**: 数据分析
- **numpy**: 数值计算
- **scikit-learn**: 机器学习工具

### 分布式计算
- **ray**: 分布式计算框架
- **networkx**: 图网络分析

## 配置要求

### 环境变量
确保 `Config/.env` 文件包含：
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
```

### 系统要求
- Python 3.9+
- Windows 10/11 或 Linux
- 推荐8GB+ RAM
- 可选：NVIDIA GPU (CUDA 11.8+)

## 运行应用

```bash
cd web_demo
python app.py
```

访问: http://localhost:5000

## 故障排除

### 常见问题

1. **torch安装失败**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

2. **sentence-transformers安装失败**
   ```bash
   pip install sentence-transformers --index-url https://pypi.org/simple/
   ```

3. **ray在Windows下问题**
   - 确保安装Visual Studio Build Tools
   - 使用最新版本ray>=2.5.0

4. **eventlet问题**
   ```bash
   pip install eventlet==0.33.3
   ```

### 依赖冲突解决
```bash
# 清理环境重新安装
pip uninstall -y torch transformers sentence-transformers
pip install -r requirements.txt
```

## 开发模式

### 安装开发依赖
```bash
pip install pytest pytest-asyncio pytest-mock black flake8
```

### 运行测试
```bash
pytest tests/
```

## 性能优化

### 可选优化包
```bash
pip install ujson cachetools redis
```

### GPU加速
确保CUDA版本匹配：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
