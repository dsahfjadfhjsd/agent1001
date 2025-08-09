"""
精简完整的配置管理系统
分离超参数、用户配置和环境变量
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()


class ConfigLoader:
    """配置文件加载器"""

    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or os.path.dirname(__file__)
        self.hyperparams = None
        self.user_profiles = None
        self._load_all_configs()

    def _load_all_configs(self):
        """加载所有配置文件"""
        # 加载超参数
        hyperparams_path = os.path.join(self.config_dir, "hyperparameters.json")
        try:
            with open(hyperparams_path, 'r', encoding='utf-8') as f:
                self.hyperparams = json.load(f)
            logging.info("超参数配置加载成功")
        except Exception as e:
            logging.warning(f"超参数配置加载失败: {e}")
            self.hyperparams = {}

        # 加载用户画像配置
        profiles_path = os.path.join(self.config_dir, "user_profiles.json")
        try:
            with open(profiles_path, 'r', encoding='utf-8') as f:
                self.user_profiles = json.load(f)
            logging.info("用户画像配置加载成功")
        except Exception as e:
            logging.warning(f"用户画像配置加载失败: {e}")
            self.user_profiles = {}

    def get_hyperparam(self, path: str, default: Any = None) -> Any:
        """获取超参数"""
        return self._get_nested_value(self.hyperparams, path, default)

    def get_user_config(self, path: str, default: Any = None) -> Any:
        """获取用户配置"""
        return self._get_nested_value(self.user_profiles, path, default)

    def _get_nested_value(self, data: Dict, path: str, default: Any) -> Any:
        """获取嵌套字典的值"""
        keys = path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def reload(self):
        """重新加载配置"""
        self._load_all_configs()


# 全局配置加载器
config_loader = ConfigLoader()


@dataclass
class ModelConfig:
    """模型配置"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model_name: str = field(default_factory=lambda: config_loader.get_hyperparam("model.model_name", "gpt-3.5-turbo"))
    temperature: float = field(default_factory=lambda: config_loader.get_hyperparam("model.temperature", 0.7))
    max_tokens: int = field(default_factory=lambda: config_loader.get_hyperparam("model.max_tokens", 1000))
    timeout: int = field(default_factory=lambda: config_loader.get_hyperparam("model.timeout", 30))


@dataclass
class ConcurrencyConfig:
    """并发配置"""
    max_concurrent_users: int = field(default_factory=lambda: config_loader.get_hyperparam("concurrency.max_concurrent_users", 10))
    batch_size: int = field(default_factory=lambda: config_loader.get_hyperparam("concurrency.batch_size", 50))
    method: str = field(default_factory=lambda: config_loader.get_hyperparam("concurrency.method", "asyncio"))
    chunk_delay: float = field(default_factory=lambda: config_loader.get_hyperparam("concurrency.chunk_delay", 0.1))
    semaphore_limit: int = field(default_factory=lambda: config_loader.get_hyperparam("concurrency.semaphore_limit", 5))


@dataclass
class SimulationConfig:
    """仿真配置"""
    max_rounds: int = field(default_factory=lambda: config_loader.get_hyperparam("simulation.max_rounds", 5))
    memory_length: int = field(default_factory=lambda: config_loader.get_hyperparam("simulation.memory_length", 5))
    activity_threshold: float = field(default_factory=lambda: config_loader.get_hyperparam("simulation.activity_threshold", 0.3))


@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics_weights: Dict[str, float] = field(default_factory=lambda: config_loader.get_hyperparam("evaluation.metrics_weights", {
        "engagement": 0.4, "reach": 0.3, "sentiment": 0.3
    }))
    recommendation_threshold: float = field(default_factory=lambda: config_loader.get_hyperparam("evaluation.recommendation_threshold", 0.6))


@dataclass
class SystemConfig:
    """系统总配置"""
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    data_path: str = field(default_factory=lambda: os.getenv("DATA_PATH", "./Data"))
    output_path: str = field(default_factory=lambda: os.getenv("OUTPUT_PATH", "./outputs"))

    model: ModelConfig = field(default_factory=ModelConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        """初始化后处理"""
        # 创建必要目录
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


# 全局配置实例
config = SystemConfig()


def get_user_generation_rules() -> Dict[str, Any]:
    """获取用户生成规则"""
    return config_loader.get_user_config("generation_rules", {})


def get_behavior_patterns() -> Dict[str, Any]:
    """获取行为模式配置"""
    return config_loader.get_user_config("behavior_patterns", {})


def get_sample_users() -> List[Dict[str, Any]]:
    """获取示例用户"""
    return config_loader.get_user_config("sample_users", [])


def reload_config():
    """重新加载配置"""
    global config, config_loader
    config_loader.reload()
    config = SystemConfig()
    return config


if __name__ == "__main__":
    # 测试配置加载
    print("🔧 配置系统测试")
    print("=" * 40)

    print(f"模型配置: {config.model.model_name}")
    print(f"并发数: {config.concurrency.max_concurrent_users}")
    print(f"并发方法: {config.concurrency.method}")
    print(f"最大轮次: {config.simulation.max_rounds}")

    print("\n用户生成规则:")
    rules = get_user_generation_rules()
    if rules:
        print(f"  年龄分布: {rules.get('age_distribution', {})}")
        print(f"  职业列表: {rules.get('occupation_list', [])[:3]}...")

    print("\n行为模式:")
    patterns = get_behavior_patterns()
    if patterns:
        print(f"  活跃度分布: {patterns.get('activity_levels', {})}")

    print(f"\n示例用户数量: {len(get_sample_users())}")

    print("\n✅ 配置加载完成!")
