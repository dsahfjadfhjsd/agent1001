"""
SimulateEnv 模块初始化文件

社交媒体交互模拟环境
提供完整的用户行为模拟和数据记录功能
"""

from .interaction_core import (
    InteractionEnvironment,
    Post,
    Comment,
    UserAction,
    ActionType
)

from .data_storage import DataStorage

from .user_behavior_simulator import (
    UserBehaviorSimulator,
    SimulationConfig
)

from .simulation_engine import SimulationEngine

__version__ = "1.0.0"
__author__ = "Agent System"

__all__ = [
    'InteractionEnvironment',
    'Post',
    'Comment',
    'UserAction',
    'ActionType',
    'DataStorage',
    'UserBehaviorSimulator',
    'SimulationConfig',
    'SimulationEngine'
]
