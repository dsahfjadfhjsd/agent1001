"""
系统配置管理模块
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class ModelConfig:
    """大模型配置"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class UserSimulationConfig:
    """用户模拟配置"""
    max_concurrent_users: int = 10
    user_memory_length: int = 5
    action_types: List[str] = field(default_factory=lambda: ["like", "comment", "share"])


@dataclass
class EnvironmentConfig:
    """环境管理配置"""
    max_rounds: int = 5
    log_level: str = "INFO"
    output_format: str = "html"
    save_path: str = "SimulateEnv/outputs"


@dataclass
class DistributionConfig:
    """分发评估配置"""
    initial_strategy: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "engagement_rate", "sentiment_shift", "opinion_diversity"
    ])


@dataclass
class SystemConfig:
    """系统总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    user_simulation: UserSimulationConfig = field(default_factory=UserSimulationConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)

    def __post_init__(self):
        """初始化后处理"""
        # 确保输出目录存在
        os.makedirs(self.environment.save_path, exist_ok=True)


# 全局配置实例
config = SystemConfig()
