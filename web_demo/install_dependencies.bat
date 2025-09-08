@echo off
echo ========================================
echo Agent1001 Web Demo 依赖安装脚本
echo ========================================

echo 检查conda环境...
conda info --envs | findstr "agent1001" >nul
if %errorlevel% neq 0 (
    echo 错误: 请先激活agent1001环境
    echo 运行: conda activate agent1001
    pause
    exit /b 1
)

echo 当前环境: %CONDA_DEFAULT_ENV%

echo.
echo 步骤1: 安装PyTorch (GPU版本)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo.
echo 步骤2: 安装核心依赖...
pip install -r requirements.txt

echo.
echo 步骤3: 验证关键包安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import flask; print(f'Flask版本: {flask.__version__}')"
python -c "import flask_socketio; print(f'Flask-SocketIO版本: {flask_socketio.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"

echo.
echo 步骤4: 检查配置文件...
if not exist "..\Config\.env" (
    echo 警告: 未找到Config/.env文件
    echo 请确保设置OPENAI_API_KEY环境变量
) else (
    echo 配置文件存在: Config/.env
)

echo.
echo ========================================
echo 安装完成！
echo 现在可以运行: python app.py
echo ========================================
pause
