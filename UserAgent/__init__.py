"""
UserAgent 用户模拟模块

提供用户画像生成和管理功能
"""

from .user_profile_generator import UserProfileGenerator
from .user_profile_manager import UserProfileManager
from .user_memory_manager import UserMemoryManager, UserThinking, UserMemoryRecord

__all__ = ['UserProfileGenerator', 'UserProfileManager', 'UserMemoryManager', 'UserThinking', 'UserMemoryRecord']

__version__ = '1.1.0'
