"""
用户记忆管理器

管理每个用户的交互历史记录，支持认知动态变化追踪
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class UserThinking:
    """用户思考记录"""
    timestamp: str
    round_number: int
    content_seen: List[str]  # 用户看到的内容（帖子、评论等）
    thinking_process: str    # 用户的思考过程
    stance_before: float     # 行为前的立场值
    stance_after: float      # 行为后的立场值
    sentiment_before: float  # 行为前的情感值
    sentiment_after: float   # 行为后的情感值
    action_taken: Optional[str] = None  # 采取的行动
    action_content: Optional[str] = None  # 行动内容（如评论文本）


@dataclass
class UserMemoryRecord:
    """用户记忆记录"""
    user_id: str
    original_profile: Dict[str, Any]  # 原始用户画像
    current_stance_value: float       # 当前立场值
    current_sentiment_value: float    # 当前情感值
    interaction_history: List[UserThinking]  # 交互历史
    last_updated: str


class UserMemoryManager:
    """用户记忆管理器"""

    def __init__(self, memory_dir: str = None):
        """
        初始化记忆管理器

        Args:
            memory_dir: 记忆文件存储目录
        """
        if memory_dir is None:
            memory_dir = os.path.join(os.path.dirname(__file__), 'user_memories')

        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # 内存缓存
        self._memory_cache: Dict[str, UserMemoryRecord] = {}

    def initialize_user_memory(self, user_profile: Dict[str, Any]) -> str:
        """
        初始化用户记忆

        Args:
            user_profile: 用户画像

        Returns:
            用户ID
        """
        user_id = user_profile['user_id']

        # 创建记忆记录
        memory_record = UserMemoryRecord(
            user_id=user_id,
            original_profile=user_profile.copy(),
            current_stance_value=float(user_profile.get('stance_value', 0.0)),
            current_sentiment_value=float(user_profile.get('sentiment_value', 0.0)),
            interaction_history=[],
            last_updated=datetime.now().isoformat()
        )

        # 保存到缓存和文件
        self._memory_cache[user_id] = memory_record
        self._save_user_memory(memory_record)

        return user_id

    def get_user_current_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取用户当前的画像（包含更新后的立场和情感值）

        Args:
            user_id: 用户ID

        Returns:
            当前用户画像，如果用户不存在返回None
        """
        memory_record = self._get_user_memory(user_id)
        if not memory_record:
            return None

        # 基于原始画像创建当前画像
        current_profile = memory_record.original_profile.copy()
        current_profile['stance_value'] = memory_record.current_stance_value
        current_profile['sentiment_value'] = memory_record.current_sentiment_value

        # 更新立场和情感的文字描述
        current_profile['stance'] = self._value_to_stance_text(memory_record.current_stance_value)
        current_profile['sentiment'] = self._value_to_sentiment_text(memory_record.current_sentiment_value)

        return current_profile

    def get_user_recent_interactions(self, user_id: str, max_count: int = 5) -> List[UserThinking]:
        """
        获取用户最近的交互记录

        Args:
            user_id: 用户ID
            max_count: 最大返回数量

        Returns:
            最近的交互记录列表
        """
        memory_record = self._get_user_memory(user_id)
        if not memory_record:
            return []

        # 返回最近的交互记录
        return memory_record.interaction_history[-max_count:]

    def add_user_thinking(self, user_id: str, thinking: UserThinking) -> bool:
        """
        添加用户思考记录

        Args:
            user_id: 用户ID
            thinking: 思考记录

        Returns:
            是否添加成功
        """
        memory_record = self._get_user_memory(user_id)
        if not memory_record:
            return False

        # 添加思考记录
        memory_record.interaction_history.append(thinking)

        # 更新当前立场和情感值
        memory_record.current_stance_value = thinking.stance_after
        memory_record.current_sentiment_value = thinking.sentiment_after
        memory_record.last_updated = datetime.now().isoformat()

        # 保存到文件
        self._save_user_memory(memory_record)

        return True

    def get_user_value_changes(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户立场和情感值的变化历史

        Args:
            user_id: 用户ID

        Returns:
            变化历史数据
        """
        memory_record = self._get_user_memory(user_id)
        if not memory_record:
            return {}

        changes = {
            'user_id': user_id,
            'original_stance': float(memory_record.original_profile.get('stance_value', 0.0)),
            'original_sentiment': float(memory_record.original_profile.get('sentiment_value', 0.0)),
            'current_stance': memory_record.current_stance_value,
            'current_sentiment': memory_record.current_sentiment_value,
            'stance_change': memory_record.current_stance_value - float(memory_record.original_profile.get('stance_value', 0.0)),
            'sentiment_change': memory_record.current_sentiment_value - float(memory_record.original_profile.get('sentiment_value', 0.0)),
            'interaction_count': len(memory_record.interaction_history),
            'stance_history': [t.stance_after for t in memory_record.interaction_history],
            'sentiment_history': [t.sentiment_after for t in memory_record.interaction_history]
        }

        return changes

    def _get_user_memory(self, user_id: str) -> Optional[UserMemoryRecord]:
        """获取用户记忆记录"""
        # 先从缓存获取
        if user_id in self._memory_cache:
            return self._memory_cache[user_id]

        # 从文件加载
        memory_file = self.memory_dir / f"{user_id}_memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 重建交互历史
                interaction_history = []
                for item in data.get('interaction_history', []):
                    thinking = UserThinking(**item)
                    interaction_history.append(thinking)

                memory_record = UserMemoryRecord(
                    user_id=data['user_id'],
                    original_profile=data['original_profile'],
                    current_stance_value=float(data['current_stance_value']),
                    current_sentiment_value=float(data['current_sentiment_value']),
                    interaction_history=interaction_history,
                    last_updated=data['last_updated']
                )

                # 加入缓存
                self._memory_cache[user_id] = memory_record
                return memory_record

            except Exception as e:
                print(f"加载用户记忆文件失败 {user_id}: {e}")

        return None

    def _save_user_memory(self, memory_record: UserMemoryRecord):
        """保存用户记忆记录"""
        memory_file = self.memory_dir / f"{memory_record.user_id}_memory.json"

        # 转换为可序列化的格式
        data = {
            'user_id': memory_record.user_id,
            'original_profile': memory_record.original_profile,
            'current_stance_value': memory_record.current_stance_value,
            'current_sentiment_value': memory_record.current_sentiment_value,
            'interaction_history': [asdict(thinking) for thinking in memory_record.interaction_history],
            'last_updated': memory_record.last_updated
        }

        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户记忆文件失败 {memory_record.user_id}: {e}")

    def _value_to_stance_text(self, value: float) -> str:
        """将立场值转换为文字描述"""
        if value > 0.5:
            return '支持'
        elif value < -0.5:
            return '反对'
        else:
            return '中立'

    def _value_to_sentiment_text(self, value: float) -> str:
        """将情感值转换为文字描述"""
        if value > 0.5:
            return '积极'
        elif value < -0.5:
            return '消极'
        else:
            return '中立'

    def list_all_users(self) -> List[str]:
        """列出所有有记忆记录的用户"""
        memory_files = list(self.memory_dir.glob("*_memory.json"))
        user_ids = []
        for file in memory_files:
            user_id = file.stem.replace('_memory', '')
            user_ids.append(user_id)
        return user_ids

    def get_statistics(self) -> Dict[str, Any]:
        """获取所有用户的统计信息"""
        all_users = self.list_all_users()

        stats = {
            'total_users': len(all_users),
            'average_stance_change': 0.0,
            'average_sentiment_change': 0.0,
            'most_active_user': None,
            'most_changed_stance': None,
            'most_changed_sentiment': None
        }

        if not all_users:
            return stats

        stance_changes = []
        sentiment_changes = []
        interaction_counts = []

        max_interactions = 0
        max_stance_change = 0
        max_sentiment_change = 0

        for user_id in all_users:
            changes = self.get_user_value_changes(user_id)
            if changes:
                stance_changes.append(abs(changes['stance_change']))
                sentiment_changes.append(abs(changes['sentiment_change']))
                interaction_counts.append(changes['interaction_count'])

                if changes['interaction_count'] > max_interactions:
                    max_interactions = changes['interaction_count']
                    stats['most_active_user'] = user_id

                if abs(changes['stance_change']) > max_stance_change:
                    max_stance_change = abs(changes['stance_change'])
                    stats['most_changed_stance'] = user_id

                if abs(changes['sentiment_change']) > max_sentiment_change:
                    max_sentiment_change = abs(changes['sentiment_change'])
                    stats['most_changed_sentiment'] = user_id

        if stance_changes:
            stats['average_stance_change'] = sum(stance_changes) / len(stance_changes)
        if sentiment_changes:
            stats['average_sentiment_change'] = sum(sentiment_changes) / len(sentiment_changes)

        return stats


# 示例使用
if __name__ == "__main__":
    # 创建记忆管理器
    memory_manager = UserMemoryManager()

    # 示例用户画像
    user_profile = {
        'user_id': 'test_user_001',
        'occupation': '工程师',
        'stance': '中立',
        'stance_value': 0.0,
        'sentiment': '中立',
        'sentiment_value': 0.0
    }

    # 初始化用户记忆
    memory_manager.initialize_user_memory(user_profile)

    # 添加思考记录
    thinking = UserThinking(
        timestamp=datetime.now().isoformat(),
        round_number=1,
        content_seen=["看到了一个关于小米的帖子"],
        thinking_process="这个帖子让我觉得小米的自动驾驶技术还需要改进",
        stance_before=0.0,
        stance_after=-0.2,
        sentiment_before=0.0,
        sentiment_after=-0.1,
        action_taken="comment_post",
        action_content="我觉得自动驾驶技术还需要更多测试"
    )

    memory_manager.add_user_thinking('test_user_001', thinking)

    # 获取当前画像
    current_profile = memory_manager.get_user_current_profile('test_user_001')
    print("当前用户画像:", current_profile)

    # 获取变化历史
    changes = memory_manager.get_user_value_changes('test_user_001')
    print("用户变化历史:", changes)
