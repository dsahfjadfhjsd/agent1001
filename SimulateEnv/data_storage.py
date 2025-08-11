"""
数据存储管理模块

负责将交互数据保存到CSV文件，并支持数据加载和查询
"""

import os
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    # 尝试相对导入
    from .interaction_core import InteractionEnvironment, UserAction, ActionType
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from interaction_core import InteractionEnvironment, UserAction, ActionType


class DataStorage:
    """数据存储管理器"""

    def __init__(self, base_dir: str = None):
        """
        初始化存储管理器

        Args:
            base_dir: 基础存储目录
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), 'data')

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # 只使用exports目录，避免数据冗余
        self.exports_dir = self.base_dir / 'exports'
        self.exports_dir.mkdir(exist_ok=True)

    def save_environment(self, env: InteractionEnvironment, session_id: str) -> str:
        """
        保存整个环境数据到exports目录

        Args:
            env: 交互环境实例
            session_id: 会话ID

        Returns:
            保存的目录路径
        """
        # 创建会话目录
        session_dir = self.exports_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # 保存环境状态
        self.save_environment_state(env, session_id, session_dir)

        # 保存用户行为
        self.save_user_actions(env.actions, session_id, session_dir)

        # 保存环境基本信息
        env_info = {
            'session_id': session_id,
            'post_id': env.post.post_id,
            'post_content': env.post.content,
            'total_rounds': env.current_round,
            'total_actions': len(env.actions),
            'total_comments': len(env.comments),
            'active_users': len(env.user_action_count),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

        info_path = session_dir / "session_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(env_info, f, indent=2, ensure_ascii=False)

        return str(session_dir)

    def save_environment_state(self, env: InteractionEnvironment, session_id: str, session_dir: Path) -> str:
        """
        保存环境状态快照

        Args:
            env: 交互环境实例
            session_id: 会话ID
            session_dir: 会话目录

        Returns:
            保存的文件路径
        """
        state = env.get_environment_state()

        # 保存帖子信息
        post_data = [{
            'post_id': state['post']['post_id'],
            'content': state['post']['content'],
            'author_id': state['post']['author_id'],
            'likes': state['post']['likes'],
            'created_at': state['post']['created_at'],
            'saved_at': datetime.now().isoformat()
        }]

        post_path = session_dir / "posts.csv"
        self._save_to_csv(post_path, post_data)

        # 保存评论信息
        comments_data = []
        for comment in state['primary_comments']:
            # 保存一级评论
            comments_data.append({
                'comment_id': comment['comment_id'],
                'post_id': env.post.post_id,
                'content': comment['content'],
                'author_id': comment['author_id'],
                'likes': comment['likes'],
                'parent_comment_id': '',
                'is_sub_comment': False,
                'created_at': comment['created_at'],
                'saved_at': datetime.now().isoformat()
            })

            # 保存二级评论
            for sub_comment in comment.get('sub_comments', []):
                comments_data.append({
                    'comment_id': sub_comment['comment_id'],
                    'post_id': env.post.post_id,
                    'content': sub_comment['content'],
                    'author_id': sub_comment['author_id'],
                    'likes': sub_comment['likes'],
                    'parent_comment_id': comment['comment_id'],
                    'is_sub_comment': True,
                    'created_at': sub_comment['created_at'],
                    'saved_at': datetime.now().isoformat()
                })

        if comments_data:
            comments_path = session_dir / "comments.csv"
            self._save_to_csv(comments_path, comments_data)

        return str(post_path)

    def save_user_actions(self, actions: List[UserAction], session_id: str, session_dir: Path) -> str:
        """
        保存用户行为数据

        Args:
            actions: 用户行为列表
            session_id: 会话ID
            session_dir: 会话目录

        Returns:
            保存的文件路径
        """
        if not actions:
            return ""

        actions_data = []
        for action in actions:
            action_data = {
                'user_id': action.user_id,
                'action_type': action.action_type.value,
                'target_id': action.target_id,
                'content': action.content or '',
                'round_number': action.round_number,
                'created_at': action.created_at.isoformat(),
                'saved_at': datetime.now().isoformat()
            }

            # 如果是评论相关行为，添加comment_id
            if action.action_type in [ActionType.COMMENT_POST, ActionType.COMMENT_COMMENT, ActionType.LIKE_COMMENT]:
                # 对于所有评论相关行为，comment_id就是target_id（如果是评论ID）
                action_data['comment_id'] = action.target_id if action.target_id.startswith('comment_') else ''
            else:
                action_data['comment_id'] = ''

            actions_data.append(action_data)

        actions_path = session_dir / "all_actions.csv"
        self._save_to_csv(actions_path, actions_data)

        return str(actions_path)

    def save_round_actions(self, actions: List[UserAction], session_id: str, round_number: int) -> str:
        """
        保存单轮用户行为数据

        Args:
            actions: 用户行为列表
            session_id: 会话ID
            round_number: 轮次号

        Returns:
            保存的文件路径
        """
        round_actions = [a for a in actions if a.round_number == round_number]

        if not round_actions:
            return ""

        actions_data = []
        for action in round_actions:
            action_data = {
                'round_number': round_number,
                'user_id': action.user_id,
                'action_type': action.action_type.value,
                'target_id': action.target_id,
                'content': action.content or '',
                'created_at': action.created_at.isoformat(),
                'saved_at': datetime.now().isoformat()
            }

            # 如果是评论相关行为，添加comment_id
            if action.action_type in [ActionType.COMMENT_POST, ActionType.COMMENT_COMMENT, ActionType.LIKE_COMMENT]:
                # 对于所有评论相关行为，comment_id就是target_id（如果是评论ID）
                action_data['comment_id'] = action.target_id if action.target_id.startswith('comment_') else ''
            else:
                action_data['comment_id'] = ''

            actions_data.append(action_data)

        # 保存到会话目录
        session_dir = self.exports_dir / session_id
        session_dir.mkdir(exist_ok=True)

        round_path = session_dir / f"round_{round_number}_actions.csv"
        self._save_to_csv(round_path, actions_data)

        return str(round_path)

    def load_environment(self, session_id: str) -> Optional[InteractionEnvironment]:
        """
        从文件加载环境数据

        Args:
            session_id: 会话ID

        Returns:
            交互环境实例，如果不存在则返回None
        """
        session_dir = self.exports_dir / session_id
        info_path = session_dir / "session_info.json"

        if not info_path.exists():
            return None

        # 加载环境基本信息
        with open(info_path, 'r', encoding='utf-8') as f:
            env_info = json.load(f)

        # 创建环境实例
        env = InteractionEnvironment(
            post_content=env_info['post_content'],
            post_id=env_info['post_id']
        )
        env.current_round = env_info['total_rounds']

        # 加载用户行为
        actions_path = session_dir / "all_actions.csv"
        if actions_path.exists():
            actions_df = pd.read_csv(actions_path)
            for _, row in actions_df.iterrows():
                action = UserAction(
                    action_id=row['action_id'],
                    user_id=row['user_id'],
                    action_type=ActionType(row['action_type']),
                    target_id=row['target_id'],
                    content=row['content'] if row['content'] else None,
                    created_at=datetime.fromisoformat(row['created_at']),
                    round_number=row['round_number']
                )
                env.add_action(action)

        return env

    def load_user_actions(self, session_id: str, user_id: str = None) -> List[Dict[str, Any]]:
        """
        加载用户行为数据

        Args:
            session_id: 会话ID
            user_id: 特定用户ID，如果不指定则返回所有用户

        Returns:
            用户行为数据列表
        """
        session_dir = self.exports_dir / session_id
        actions_path = session_dir / "all_actions.csv"

        if not actions_path.exists():
            return []

        df = pd.read_csv(actions_path)

        if user_id:
            df = df[df['user_id'] == user_id]

        return df.to_dict('records')

    def load_round_actions(self, session_id: str, round_number: int) -> List[Dict[str, Any]]:
        """
        加载指定轮次的行为数据

        Args:
            session_id: 会话ID
            round_number: 轮次号

        Returns:
            行为数据列表
        """
        session_dir = self.exports_dir / session_id

        # 先尝试加载单独的轮次文件
        round_path = session_dir / f"round_{round_number}_actions.csv"
        if round_path.exists():
            df = pd.read_csv(round_path)
            return df.to_dict('records')

        # 如果没有单独文件，从总文件中筛选
        actions_path = session_dir / "all_actions.csv"
        if not actions_path.exists():
            return []

        df = pd.read_csv(actions_path)
        df = df[df['round_number'] == round_number]

        return df.to_dict('records')

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话摘要信息

        Args:
            session_id: 会话ID

        Returns:
            会话摘要字典
        """
        session_dir = self.exports_dir / session_id
        info_path = session_dir / "session_info.json"

        if not info_path.exists():
            return None

        with open(info_path, 'r', encoding='utf-8') as f:
            env_info = json.load(f)

        # 加载行为统计
        actions_path = session_dir / "all_actions.csv"
        action_stats = {}
        if actions_path.exists():
            df = pd.read_csv(actions_path)
            action_stats = {
                'total_actions': len(df),
                'actions_by_type': df['action_type'].value_counts().to_dict(),
                'actions_by_round': df['round_number'].value_counts().sort_index().to_dict(),
                'unique_users': df['user_id'].nunique()
            }

        env_info.update(action_stats)
        return env_info

    def list_sessions(self) -> List[str]:
        """列出所有会话ID"""
        session_dirs = [d for d in self.exports_dir.iterdir() if d.is_dir()]
        return [d.name for d in session_dirs]

    def _save_to_csv(self, file_path: Path, data: List[Dict[str, Any]], mode: str = 'w'):
        """
        保存数据到CSV文件

        Args:
            file_path: 文件路径
            data: 数据列表
            mode: 写入模式（'w' 覆盖，'a' 追加）
        """
        if not data:
            return

        fieldnames = list(data[0].keys())

        # 如果是追加模式且文件已存在，检查表头
        write_header = mode == 'w' or not file_path.exists()

        with open(file_path, mode, newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(data)

    def export_session_data(self, session_id: str, export_dir: str = None) -> str:
        """
        获取会话数据目录路径（数据已经存储在exports目录中）

        Args:
            session_id: 会话ID
            export_dir: 导出目录，如果指定则复制到该目录

        Returns:
            导出目录路径
        """
        session_dir = self.exports_dir / session_id

        if export_dir is None:
            # 直接返回现有的会话目录路径
            return str(session_dir)
        else:
            # 如果指定了导出目录，则复制文件
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)

            # 复制所有文件
            for file_path in session_dir.glob("*"):
                if file_path.is_file():
                    dst_path = export_path / file_path.name
                    dst_path.write_bytes(file_path.read_bytes())

            return str(export_path)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append('.')
    from interaction_core import InteractionEnvironment, UserAction, ActionType

    # 创建测试环境
    env = InteractionEnvironment("测试帖子内容")
    env.start_new_round()

    # 添加一些测试行为
    action1 = UserAction("", "user1", ActionType.LIKE_POST, env.post.post_id)
    env.add_action(action1)

    action2 = UserAction("", "user2", ActionType.COMMENT_POST, env.post.post_id, "测试评论")
    env.add_action(action2)

    # 测试存储
    storage = DataStorage()
    session_id = "test_session_001"

    # 保存环境
    save_path = storage.save_environment(env, session_id)
    print(f"环境已保存到: {save_path}")

    # 获取会话摘要
    summary = storage.get_session_summary(session_id)
    print(f"会话摘要: {summary}")

    # 列出所有会话
    sessions = storage.list_sessions()
    print(f"所有会话: {sessions}")

    # 导出会话数据
    export_path = storage.export_session_data(session_id)
    print(f"数据导出路径: {export_path}")

    # 加载环境
    loaded_env = storage.load_environment(session_id)
    if loaded_env:
        print(f"成功加载环境，行为数量: {len(loaded_env.actions)}")

    # 列出所有会话
    sessions = storage.list_sessions()
    print(f"所有会话: {sessions}")
