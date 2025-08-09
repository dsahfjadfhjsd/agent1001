# 认知引导系统 (Cognitive Guidance System)

一个基于AI智能体的用户模拟和内容分发评估系统，用于研究和优化针对特定事件的认知引导效果。

## 🎯 系统概述

本系统由三个核心模块组成，实现了从用户模拟到内容分发再到效果评估的完整闭环：

### 1. 用户模拟模块 (UserAgent)

- **用户画像管理**: 支持基本社会特征（年龄、性别、职业等）和事件态度（立场、情感、意图）
- **智能体行为**: 基于画像和记忆的智能决策，支持点赞、评论、分享等行为
- **并发模拟**: 支持多用户并发交互模拟

### 2. 环境管理模块 (SimulateEnv)

- **多轮交互**: 支持多轮次的用户交互模拟
- **层级评论**: 支持帖子评论和二级回复
- **实时记录**: 完整记录用户行为和交互历史
- **可视化输出**: 生成HTML格式的交互报告

### 3. 分发评估模块 (DisAgent)

- **智能分发**: AI智能体分析内容并推荐目标用户
- **规则引擎**: 支持基于规则的用户过滤和选择
- **效果评估**: 多维度评估分发效果（参与率、情感变化、观点多样性等）
- **策略优化**: 自动生成优化建议

## 🚀 快速开始

### 环境要求

- Python 3.9+
- OpenAI API Key 或兼容的大模型服务

### 安装依赖

```bash
# 克隆项目
cd your_project_directory

# 安装依赖
uv sync
```

### 配置

1. 复制环境变量配置文件：

```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的API配置：

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 运行演示

```bash
# 激活虚拟环境（如果使用uv）
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 运行主程序
python main.py
```

### 快速模拟示例

```python
import asyncio
from cognitive_guidance_system import quick_simulation

async def main():
    result = await quick_simulation(
        post_content="人工智能对未来工作的影响讨论",
        post_title="AI与就业",
        max_rounds=3
    )
    print(f"参与率: {result['final_metrics']['engagement_rate']:.2%}")

asyncio.run(main())
```

## 📊 系统架构

```
认知引导系统
├── UserAgent/              # 用户模拟模块
│   ├── user_profile.py     # 用户画像定义
│   └── user_agent.py       # 用户智能体实现
├── SimulateEnv/            # 环境管理模块
│   ├── environment_models.py  # 环境数据模型
│   └── environment_manager.py # 环境管理器
├── DisAgent/               # 分发评估模块
│   ├── distribution_models.py # 分发数据模型
│   └── distribution_agent.py  # 分发智能体
├── Config/                 # 配置管理
│   ├── settings.py         # 系统配置
│   └── sample_users.json   # 示例用户数据
└── cognitive_guidance_system.py # 系统集成
```

## 🔧 核心功能

### 1. 用户画像定义

```python
from UserAgent.user_profile import UserProfile, Gender, Stance, Emotion, Intent

profile = UserProfile(
    user_id="user_001",
    age=28,
    gender=Gender.FEMALE,
    occupation="教师",
    stance=Stance.SUPPORT,
    emotion=Emotion.POSITIVE,
    intent=Intent.DISCUSS,
    activity_level=0.7,
    social_influence=0.6
)
```

### 2. 环境模拟

```python
from cognitive_guidance_system import CognitiveGuidanceSystem

system = CognitiveGuidanceSystem()
system.initialize_system()

post_data = {
    "title": "讨论主题",
    "content": "帖子内容",
    "author": "发布者"
}

result = await system.start_simulation(post_data, max_rounds=3)
```

### 3. 分发策略配置

```python
from DisAgent.distribution_models import DistributionStrategy, DistributionRule

strategy = DistributionStrategy(
    strategy_id="custom_strategy",
    name="自定义策略",
    description="基于特定条件的分发策略"
)

# 添加规则
rule = DistributionRule(
    rule_id="activity_rule",
    name="活跃用户优先",
    conditions={"activity_level": 0.5},
    weight=1.0
)
strategy.add_rule(rule)
```

## 📈 评估指标

系统提供多维度的效果评估：

- **参与率** (Engagement Rate): 用户交互比例
- **情感变化** (Sentiment Shift): 评论情感倾向变化
- **观点多样性** (Opinion Diversity): 评论内容多样性
- **触达范围** (Reach): 实际参与用户数
- **转化率** (Conversion Rate): 有意义交互比例
- **病毒性评分** (Virality Score): 内容传播潜力

## 📁 输出文件

模拟完成后，系统会在 `SimulateEnv/outputs/` 目录生成：

- `session_*.json`: 完整的模拟数据
- `report_*.html`: 可视化交互报告
- `simulation_report_*.json`: 综合分析报告

## 🛠️ 自定义配置

### 修改用户配置

编辑 `Config/sample_users.json` 文件来自定义用户画像：

```json
{
    "user_id": "custom_user",
    "age": 30,
    "gender": "male",
    "occupation": "工程师",
    "stance": "neutral",
    "emotion": "positive",
    "intent": "discuss",
    "activity_level": 0.8
}
```

### 调整系统参数

修改 `Config/settings.py` 中的配置：

```python
@dataclass
class UserSimulationConfig:
    max_concurrent_users: int = 20  # 最大并发用户数
    user_memory_length: int = 10    # 用户记忆长度
    action_types: List[str] = ["like", "comment", "share", "forward"]
```

## 🔄 扩展开发

### 添加新的行为类型

1. 在 `UserAgent/user_agent.py` 中扩展 `UserAction` 类
2. 在 `SimulateEnv/environment_models.py` 中添加对应的数据模型
3. 在环境管理器中添加处理逻辑

### 自定义评估指标

1. 在 `DisAgent/distribution_models.py` 中扩展 `EvaluationMetrics`
2. 在 `DisAgent/distribution_agent.py` 中实现计算逻辑

### 集成外部数据

系统支持从现有的社交媒体数据中加载：

- 参考 `Data/` 目录中的数据结构
- 实现相应的数据转换器

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request来改进系统功能。

## 📞 联系

如有问题或建议，请通过Issue与我们联系。
