"""
交互环境核心模块

管理帖子、评论、用户行为等核心数据结构
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    """用户行为类型"""
    LIKE_POST = "like_post"
    COMMENT_POST = "comment_post"
    LIKE_COMMENT = "like_comment"
    COMMENT_COMMENT = "comment_comment"  # 二级评论


@dataclass
class Post:
    """帖子数据结构"""
    post_id: str
    content: str
    author_id: str = "original_author"
    created_at: datetime = field(default_factory=datetime.now)
    likes: int = 0
    like_users: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.post_id:
            self.post_id = f"post_{uuid.uuid4().hex[:8]}"


@dataclass
class Comment:
    """评论数据结构"""
    comment_id: str
    post_id: str
    content: str
    author_id: str
    parent_comment_id: Optional[str] = None  # 父评论ID，用于二级评论
    created_at: datetime = field(default_factory=datetime.now)
    likes: int = 0
    like_users: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.comment_id:
            if self.parent_comment_id:
                # 二级评论：父评论ID_子评论ID
                self.comment_id = f"{self.parent_comment_id}_sub_{uuid.uuid4().hex[:6]}"
            else:
                # 一级评论：帖子ID_评论ID
                self.comment_id = f"{self.post_id}_comment_{uuid.uuid4().hex[:6]}"

    @property
    def is_sub_comment(self) -> bool:
        """是否为二级评论"""
        return self.parent_comment_id is not None


@dataclass
class UserAction:
    """用户行为记录"""
    action_id: str
    user_id: str
    action_type: ActionType
    target_id: str  # 目标ID（帖子ID或评论ID）
    content: Optional[str] = None  # 评论内容
    created_at: datetime = field(default_factory=datetime.now)
    round_number: int = 1

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"action_{uuid.uuid4().hex[:8]}"


class InteractionEnvironment:
    """交互环境管理器"""

    def __init__(self, post_content: str, post_id: str = None):
        """
        初始化交互环境

        Args:
            post_content: 初始帖子内容
            post_id: 帖子ID，如果不提供则自动生成
        """
        self.post = Post(
            post_id=post_id or f"post_{uuid.uuid4().hex[:8]}",
            content=post_content
        )
        self.comments: Dict[str, Comment] = {}  # comment_id -> Comment
        self.actions: List[UserAction] = []  # 所有用户行为记录
        self.current_round = 0
        self.user_action_count: Dict[str, int] = {}  # 用户行为统计

    def add_action(self, action: UserAction) -> bool:
        """
        添加用户行为

        Args:
            action: 用户行为

        Returns:
            是否添加成功
        """
        try:
            # 根据行为类型执行相应操作
            if action.action_type == ActionType.LIKE_POST:
                self._like_post(action.user_id, action.target_id)
            elif action.action_type == ActionType.COMMENT_POST:
                self._comment_post(action.user_id, action.target_id, action.content)
            elif action.action_type == ActionType.LIKE_COMMENT:
                self._like_comment(action.user_id, action.target_id)
            elif action.action_type == ActionType.COMMENT_COMMENT:
                self._comment_comment(action.user_id, action.target_id, action.content)

            # 记录行为
            action.round_number = self.current_round
            self.actions.append(action)

            # 更新用户行为统计
            self.user_action_count[action.user_id] = self.user_action_count.get(action.user_id, 0) + 1

            return True

        except Exception as e:
            print(f"添加行为失败: {e}")
            return False

    def _like_post(self, user_id: str, post_id: str):
        """点赞帖子"""
        if post_id != self.post.post_id:
            raise ValueError(f"帖子ID不匹配: {post_id}")

        if user_id not in self.post.like_users:
            self.post.like_users.append(user_id)
            self.post.likes += 1

    def _comment_post(self, user_id: str, post_id: str, content: str):
        """评论帖子"""
        if post_id != self.post.post_id:
            raise ValueError(f"帖子ID不匹配: {post_id}")

        comment = Comment(
            comment_id="",  # 会在__post_init__中生成
            post_id=post_id,
            content=content,
            author_id=user_id
        )
        self.comments[comment.comment_id] = comment

    def _like_comment(self, user_id: str, comment_id: str):
        """点赞评论"""
        if comment_id not in self.comments:
            raise ValueError(f"评论不存在: {comment_id}")

        comment = self.comments[comment_id]
        if user_id not in comment.like_users:
            comment.like_users.append(user_id)
            comment.likes += 1

    def _comment_comment(self, user_id: str, parent_comment_id: str, content: str):
        """回复评论（二级评论）"""
        if parent_comment_id not in self.comments:
            raise ValueError(f"父评论不存在: {parent_comment_id}")

        parent_comment = self.comments[parent_comment_id]
        sub_comment = Comment(
            comment_id="",  # 会在__post_init__中生成
            post_id=parent_comment.post_id,
            content=content,
            author_id=user_id,
            parent_comment_id=parent_comment_id
        )
        self.comments[sub_comment.comment_id] = sub_comment

    def get_environment_state(self) -> Dict[str, Any]:
        """
        获取当前环境状态

        Returns:
            环境状态字典
        """
        # 获取一级评论
        primary_comments = [c for c in self.comments.values() if not c.is_sub_comment]
        primary_comments.sort(key=lambda x: x.created_at)

        # 获取二级评论，按父评论分组
        sub_comments_by_parent = {}
        for comment in self.comments.values():
            if comment.is_sub_comment:
                parent_id = comment.parent_comment_id
                if parent_id not in sub_comments_by_parent:
                    sub_comments_by_parent[parent_id] = []
                sub_comments_by_parent[parent_id].append(comment)

        # 排序二级评论
        for parent_id in sub_comments_by_parent:
            sub_comments_by_parent[parent_id].sort(key=lambda x: x.created_at)

        return {
            'post': {
                'post_id': self.post.post_id,
                'content': self.post.content,
                'author_id': self.post.author_id,
                'likes': self.post.likes,
                'created_at': self.post.created_at.isoformat()
            },
            'primary_comments': [
                {
                    'comment_id': c.comment_id,
                    'content': c.content,
                    'author_id': c.author_id,
                    'likes': c.likes,
                    'created_at': c.created_at.isoformat(),
                    'sub_comments': [
                        {
                            'comment_id': sub.comment_id,
                            'content': sub.content,
                            'author_id': sub.author_id,
                            'likes': sub.likes,
                            'created_at': sub.created_at.isoformat()
                        }
                        for sub in sub_comments_by_parent.get(c.comment_id, [])
                    ]
                }
                for c in primary_comments
            ],
            'total_comments': len(self.comments),
            'total_actions': len(self.actions),
            'current_round': self.current_round,
            'active_users': len(self.user_action_count)
        }

    def get_available_targets(self) -> Dict[str, List[str]]:
        """
        获取可操作的目标列表

        Returns:
            可操作目标字典
        """
        return {
            'post_id': self.post.post_id,
            'comment_ids': list(self.comments.keys()),
            'primary_comment_ids': [c.comment_id for c in self.comments.values() if not c.is_sub_comment],
            'sub_comment_ids': [c.comment_id for c in self.comments.values() if c.is_sub_comment]
        }

    def start_new_round(self):
        """开始新一轮交互"""
        self.current_round += 1

    def get_user_actions(self, user_id: str) -> List[UserAction]:
        """获取指定用户的所有行为"""
        return [action for action in self.actions if action.user_id == user_id]

    def get_round_actions(self, round_number: int) -> List[UserAction]:
        """获取指定轮次的所有行为"""
        return [action for action in self.actions if action.round_number == round_number]


if __name__ == "__main__":
    # 测试代码
    env = InteractionEnvironment("这是一个测试帖子的内容")

    # 添加一些测试行为
    env.start_new_round()

    # 用户1点赞帖子
    action1 = UserAction("", "user1", ActionType.LIKE_POST, env.post.post_id)
    env.add_action(action1)

    # 用户2评论帖子
    action2 = UserAction("", "user2", ActionType.COMMENT_POST, env.post.post_id, "这是第一条评论")
    env.add_action(action2)

    # 获取评论ID进行后续操作
    comment_ids = list(env.comments.keys())
    if comment_ids:
        # 用户3点赞评论
        action3 = UserAction("", "user3", ActionType.LIKE_COMMENT, comment_ids[0])
        env.add_action(action3)

        # 用户1回复评论
        action4 = UserAction("", "user1", ActionType.COMMENT_COMMENT, comment_ids[0], "这是回复评论")
        env.add_action(action4)

    # 打印环境状态
    import json
    state = env.get_environment_state()
    print(json.dumps(state, indent=2, ensure_ascii=False))
