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
            # 使用项目根目录下的Output文件夹
            project_root = os.path.dirname(os.path.dirname(__file__))
            base_dir = os.path.join(project_root, 'Output')

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # 使用exports目录
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

    def save_environment_state(self, env: InteractionEnvironment, post_id: str, session_dir: Path) -> str:
        """
        保存环境状态快照

        Args:
            env: 交互环境实例
            post_id: 帖子ID
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

    def save_user_actions(self, actions: List[UserAction], post_id: str, session_dir: Path) -> str:
        """
        保存用户行为数据

        Args:
            actions: 用户行为列表
            post_id: 帖子ID
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
                # 优先使用action.comment_id，如果没有则从target_id推断
                if hasattr(action, 'comment_id') and action.comment_id:
                    action_data['comment_id'] = action.comment_id
                else:
                    action_data['comment_id'] = action.target_id if action.target_id.startswith('comment_') else ''
            else:
                action_data['comment_id'] = ''

            actions_data.append(action_data)

        actions_path = session_dir / "all_actions.csv"
        self._save_to_csv(actions_path, actions_data)

        return str(actions_path)

    def save_incremental_data(self, env: InteractionEnvironment, post_id: str, batch_id: str = None) -> str:
        """
        增量保存环境数据（每轮结束后调用）

        Args:
            env: 交互环境实例
            post_id: 帖子ID（作为主要标识符）
            batch_id: 批次ID（可选，用于进一步分组）

        Returns:
            保存的目录路径
        """
        # 创建目录结构：如果有batch_id则是 batch_id/post_id，否则直接是post_id
        if batch_id:
            session_dir = self.exports_dir / batch_id / post_id
        else:
            session_dir = self.exports_dir / post_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 保存环境状态（覆盖保存）
        self.save_environment_state(env, post_id, session_dir)

        # 增量保存用户行为（追加模式）
        self._save_user_actions_incremental(env.actions, session_dir)

        # 保存/更新环境基本信息
        env_info = {
            'post_id': post_id,  # 使用post_id作为主要标识符
            'batch_id': batch_id,  # 记录batch_id（如果有的话）
            'post_content': env.post.content,
            'total_rounds': env.current_round,
            'total_actions': len(env.actions),
            'total_comments': len(env.comments),
            'active_users': len(env.user_action_count),
            'created_at': getattr(self, '_session_created_at', datetime.now().isoformat()),
            'last_updated': datetime.now().isoformat()
        }

        # 记录会话创建时间（仅第一次）
        if not hasattr(self, '_session_created_at'):
            self._session_created_at = env_info['created_at']

        info_path = session_dir / "session_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(env_info, f, indent=2, ensure_ascii=False)

        return str(session_dir)

    def _save_user_actions_incremental(self, actions: List[UserAction], session_dir: Path):
        """
        增量保存用户行为数据

        Args:
            actions: 用户行为列表
            session_dir: 会话目录
        """
        if not actions:
            return

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
                # 优先使用action.comment_id，如果没有则从target_id推断
                if hasattr(action, 'comment_id') and action.comment_id:
                    action_data['comment_id'] = action.comment_id
                else:
                    action_data['comment_id'] = action.target_id if action.target_id.startswith('comment_') else ''
            else:
                action_data['comment_id'] = ''

            actions_data.append(action_data)

        actions_path = session_dir / "all_actions.csv"
        # 追加保存，保持所有轮次的行为记录
        self._save_to_csv(actions_path, actions_data, mode='a')

    def load_environment(self, post_id: str) -> Optional[InteractionEnvironment]:
        """
        从文件加载环境数据

        Args:
            post_id: 帖子ID

        Returns:
            交互环境实例，如果不存在则返回None
        """
        # 首先尝试直接查找
        session_dir = self.exports_dir / post_id
        info_path = session_dir / "session_info.json"

        # 如果直接查找不到，尝试在所有batch目录中查找
        if not info_path.exists():
            for batch_dir in self.exports_dir.iterdir():
                if batch_dir.is_dir():
                    potential_path = batch_dir / post_id / "session_info.json"
                    if potential_path.exists():
                        session_dir = batch_dir / post_id
                        info_path = potential_path
                        break
            else:
                return None

        # 加载环境基本信息
        with open(info_path, 'r', encoding='utf-8') as f:
            env_info = json.load(f)

        # 创建环境实例
        env = InteractionEnvironment(
            post_content=env_info['post_content'],
            post_id=env_info.get('post_id', post_id)
        )
        env.current_round = env_info['total_rounds']

        # 加载用户行为
        actions_path = session_dir / "all_actions.csv"
        if actions_path.exists():
            actions_df = pd.read_csv(actions_path)
            for _, row in actions_df.iterrows():
                action = UserAction(
                    action_id=row.get('action_id', ''),
                    user_id=row['user_id'],
                    action_type=ActionType(row['action_type']),
                    target_id=row['target_id'],
                    content=row['content'] if row['content'] else None,
                    created_at=datetime.fromisoformat(row['created_at']),
                    round_number=row['round_number']
                )
                env.add_action(action)

        return env

    def load_user_actions(self, post_id: str, user_id: str = None) -> List[Dict[str, Any]]:
        """
        加载用户行为数据

        Args:
            post_id: 帖子ID
            user_id: 特定用户ID，如果不指定则返回所有用户

        Returns:
            用户行为数据列表
        """
        # 查找post_id对应的目录
        session_dir = self._find_post_directory(post_id)
        if not session_dir:
            return []

        actions_path = session_dir / "all_actions.csv"
        if not actions_path.exists():
            return []

        df = pd.read_csv(actions_path)

        if user_id:
            df = df[df['user_id'] == user_id]

        return df.to_dict('records')

    def load_round_actions(self, post_id: str, round_number: int) -> List[Dict[str, Any]]:
        """
        加载指定轮次的行为数据

        Args:
            post_id: 帖子ID
            round_number: 轮次号

        Returns:
            行为数据列表
        """
        # 查找post_id对应的目录
        session_dir = self._find_post_directory(post_id)
        if not session_dir:
            return []

        actions_path = session_dir / "all_actions.csv"
        if not actions_path.exists():
            return []

        df = pd.read_csv(actions_path)
        df = df[df['round_number'] == round_number]

        return df.to_dict('records')

    def get_session_summary(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话摘要信息

        Args:
            post_id: 帖子ID

        Returns:
            会话摘要字典
        """
        # 查找post_id对应的目录
        session_dir = self._find_post_directory(post_id)
        if not session_dir:
            return None

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
        """列出所有post_id"""
        post_ids = []

        # 查找根目录下的post_id
        for item in self.exports_dir.iterdir():
            if item.is_dir():
                info_path = item / "session_info.json"
                if info_path.exists():
                    post_ids.append(item.name)
                else:
                    # 检查是否是batch目录
                    for sub_item in item.iterdir():
                        if sub_item.is_dir():
                            sub_info_path = sub_item / "session_info.json"
                            if sub_info_path.exists():
                                post_ids.append(sub_item.name)

        return list(set(post_ids))  # 去重

    def _find_post_directory(self, post_id: str) -> Optional[Path]:
        """
        查找post_id对应的目录

        Args:
            post_id: 帖子ID

        Returns:
            目录路径，如果不存在返回None
        """
        # 首先尝试直接查找
        session_dir = self.exports_dir / post_id
        if (session_dir / "session_info.json").exists():
            return session_dir

        # 在所有batch目录中查找
        for batch_dir in self.exports_dir.iterdir():
            if batch_dir.is_dir():
                potential_dir = batch_dir / post_id
                if (potential_dir / "session_info.json").exists():
                    return potential_dir

        return None

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

    def export_session_data(self, post_id: str, export_dir: str = None) -> str:
        """
        获取会话数据目录路径（数据已经存储在exports目录中）

        Args:
            post_id: 帖子ID
            export_dir: 导出目录，如果指定则复制到该目录

        Returns:
            导出目录路径
        """
        session_dir = self._find_post_directory(post_id)
        if not session_dir:
            raise FileNotFoundError(f"找不到post_id为 {post_id} 的会话数据")

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
