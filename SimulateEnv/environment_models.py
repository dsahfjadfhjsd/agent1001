"""
环境内容数据结构定义
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


@dataclass
class Post:
    """帖子内容"""
    post_id: str
    title: str
    content: str
    author: str
    timestamp: datetime
    tags: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)  # 图片、视频等媒体文件

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_id": self.post_id,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "media_urls": self.media_urls
        }


@dataclass
class Comment:
    """评论"""
    comment_id: str
    post_id: str
    parent_comment_id: Optional[str]  # 父评论ID，用于二级评论
    user_id: str
    content: str
    timestamp: datetime
    likes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comment_id": self.comment_id,
            "post_id": self.post_id,
            "parent_comment_id": self.parent_comment_id,
            "user_id": self.user_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "likes": self.likes
        }


@dataclass
class Like:
    """点赞"""
    like_id: str
    target_id: str  # 可以是帖子ID或评论ID
    target_type: str  # post 或 comment
    user_id: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "like_id": self.like_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Share:
    """分享"""
    share_id: str
    post_id: str
    user_id: str
    timestamp: datetime
    share_content: str = ""  # 分享时的附加内容

    def to_dict(self) -> Dict[str, Any]:
        return {
            "share_id": self.share_id,
            "post_id": self.post_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "share_content": self.share_content
        }


@dataclass
class EnvironmentState:
    """环境状态"""
    round_number: int
    post: Post
    comments: List[Comment] = field(default_factory=list)
    likes: List[Like] = field(default_factory=list)
    shares: List[Share] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)  # 参与用户ID列表

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "post": self.post.to_dict(),
            "comments": [comment.to_dict() for comment in self.comments],
            "likes": [like.to_dict() for like in self.likes],
            "shares": [share.to_dict() for share in self.shares],
            "participants": self.participants
        }

    def get_comments_by_post(self, post_id: str) -> List[Comment]:
        """获取帖子的所有评论"""
        return [comment for comment in self.comments if comment.post_id == post_id and comment.parent_comment_id is None]

    def get_replies_by_comment(self, comment_id: str) -> List[Comment]:
        """获取评论的所有回复"""
        return [comment for comment in self.comments if comment.parent_comment_id == comment_id]

    def get_likes_by_target(self, target_id: str) -> List[Like]:
        """获取目标对象的所有点赞"""
        return [like for like in self.likes if like.target_id == target_id]

    def get_engagement_stats(self) -> Dict[str, int]:
        """获取互动统计"""
        return {
            "total_comments": len(self.comments),
            "total_likes": len(self.likes),
            "total_shares": len(self.shares),
            "unique_participants": len(set(self.participants))
        }
