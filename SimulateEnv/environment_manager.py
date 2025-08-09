"""
环境管理器核心实现
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import os

from .environment_models import Post, Comment, Like, Share, EnvironmentState
from UserAgent.user_agent import UserAgent, UserAction, UserAgentManager
from Config.settings import config


class EnvironmentManager:
    """环境管理器"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.current_state: Optional[EnvironmentState] = None
        self.history: List[EnvironmentState] = []
        self.user_manager = UserAgentManager()

        # 设置日志
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f"EnvironmentManager_{self.session_id}")
        logger.setLevel(getattr(logging, config.environment.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def initialize_environment(self, post: Post) -> EnvironmentState:
        """初始化环境"""
        self.current_state = EnvironmentState(
            round_number=0,
            post=post,
            comments=[],
            likes=[],
            shares=[],
            participants=[]
        )
        self.history.append(self.current_state)
        self.logger.info(f"环境已初始化，帖子ID: {post.post_id}")
        return self.current_state

    def _process_user_action(self, action: UserAction, user_id: str):
        """处理用户动作"""
        timestamp = datetime.now()

        if action.action_type == "like":
            like = Like(
                like_id=str(uuid.uuid4()),
                target_id=action.target_id,
                target_type="post",  # 可以根据target_id判断类型
                user_id=user_id,
                timestamp=timestamp
            )
            self.current_state.likes.append(like)

        elif action.action_type == "comment":
            comment = Comment(
                comment_id=str(uuid.uuid4()),
                post_id=self.current_state.post.post_id,
                parent_comment_id=None if action.target_id == self.current_state.post.post_id else action.target_id,
                user_id=user_id,
                content=action.content,
                timestamp=timestamp
            )
            self.current_state.comments.append(comment)

        elif action.action_type == "share":
            share = Share(
                share_id=str(uuid.uuid4()),
                post_id=self.current_state.post.post_id,
                user_id=user_id,
                timestamp=timestamp,
                share_content=action.content
            )
            self.current_state.shares.append(share)

        # 添加到参与者列表
        if user_id not in self.current_state.participants:
            self.current_state.participants.append(user_id)

    async def simulate_round(self, participating_users: List[str], concurrent: bool = True,
                             concurrency_method: str = "asyncio") -> Dict[str, Any]:
        """模拟一轮交互

        Args:
            participating_users: 参与用户列表
            concurrent: 是否启用并发
            concurrency_method: 并发方法 ("asyncio", "futures", "chunked")
        """
        if not self.current_state:
            raise ValueError("环境未初始化，请先调用 initialize_environment")

        round_start_time = datetime.now()
        self.current_state.round_number += 1
        current_round = self.current_state.round_number

        self.logger.info(f"开始第 {current_round} 轮模拟，参与用户数: {len(participating_users)}, 并发方法: {concurrency_method}")

        # 构建环境内容供用户决策
        environment_content = {
            "post_id": self.current_state.post.post_id,
            "post": self.current_state.post.to_dict(),
            "comments": [comment.to_dict() for comment in self.current_state.comments],
            "likes_count": len(self.current_state.likes),
            "shares_count": len(self.current_state.shares),
            "round": current_round
        }

        if concurrent:
            # 根据方法选择并发实现
            if concurrency_method == "asyncio":
                actions = await self.user_manager.simulate_batch_actions(
                    participating_users, environment_content
                )
            elif concurrency_method == "futures":
                actions = await self.user_manager.simulate_batch_actions_with_futures(
                    participating_users, environment_content
                )
            elif concurrency_method == "chunked":
                actions = await self.user_manager.simulate_batch_actions_chunked(
                    participating_users, environment_content
                )
            else:
                self.logger.warning(f"未知的并发方法: {concurrency_method}，使用默认asyncio方法")
                actions = await self.user_manager.simulate_batch_actions(
                    participating_users, environment_content
                )
        else:
            # 串行模拟
            actions = []
            for user_id in participating_users:
                agent = self.user_manager.get_agent(user_id)
                if agent:
                    action = await agent.decide_action(environment_content)
                    if action:
                        actions.append(action)

        # 处理所有用户动作
        for action in actions:
            # 找到对应的用户ID
            user_id = None
            for uid in participating_users:
                agent = self.user_manager.get_agent(uid)
                if agent and agent.memory.interactions and agent.memory.interactions[-1]["action"]["action_id"] == action.action_id:
                    user_id = uid
                    break

            if user_id:
                self._process_user_action(action, user_id)

        # 记录本轮结果
        round_summary = {
            "round": current_round,
            "duration": (datetime.now() - round_start_time).total_seconds(),
            "participating_users": len(participating_users),
            "actions_taken": len(actions),
            "action_types": {action.action_type: sum(1 for a in actions if a.action_type == action.action_type)
                             for action in actions},
            "engagement_stats": self.current_state.get_engagement_stats()
        }

        self.logger.info(f"第 {current_round} 轮完成: {round_summary}")

        # 保存当前状态到历史
        self.history.append(self.current_state)

        return round_summary

    def get_environment_for_display(self) -> Dict[str, Any]:
        """获取用于展示的环境数据"""
        if not self.current_state:
            return {}

        # 构建展示数据结构，类似社交媒体界面
        display_data = {
            "session_id": self.session_id,
            "post": self.current_state.post.to_dict(),
            "stats": {
                "likes": len(self.current_state.likes),
                "comments": len(self.current_state.comments),
                "shares": len(self.current_state.shares)
            },
            "comments": []
        }

        # 组织评论和回复的层级结构
        top_level_comments = self.current_state.get_comments_by_post(self.current_state.post.post_id)

        for comment in top_level_comments:
            comment_data = comment.to_dict()
            comment_data["likes_count"] = len(self.current_state.get_likes_by_target(comment.comment_id))
            comment_data["replies"] = []

            # 获取该评论的回复
            replies = self.current_state.get_replies_by_comment(comment.comment_id)
            for reply in replies:
                reply_data = reply.to_dict()
                reply_data["likes_count"] = len(self.current_state.get_likes_by_target(reply.comment_id))
                comment_data["replies"].append(reply_data)

            display_data["comments"].append(comment_data)

        return display_data

    def save_session(self, filename: str = None):
        """保存会话数据"""
        if filename is None:
            filename = f"session_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(config.environment.save_path, filename)

        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_rounds": len(self.history),
            "final_state": self.current_state.to_dict() if self.current_state else None,
            "display_data": self.get_environment_for_display()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"会话数据已保存到: {filepath}")
        return filepath

    def load_users(self, user_config_path: str):
        """加载用户配置"""
        self.user_manager.load_agents_from_config(user_config_path)
        self.logger.info(f"已加载 {len(self.user_manager.agents)} 个用户智能体")

    def export_html_report(self, filename: str = None) -> str:
        """导出HTML格式的报告"""
        if filename is None:
            filename = f"report_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        filepath = os.path.join(config.environment.save_path, filename)
        display_data = self.get_environment_for_display()

        html_content = self._generate_html_report(display_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"HTML报告已导出到: {filepath}")
        return filepath

    def _generate_html_report(self, display_data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>用户交互模拟报告</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .post { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .post-header { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .post-content { margin-bottom: 15px; line-height: 1.6; }
        .post-stats { color: #666; font-size: 14px; margin-bottom: 20px; }
        .comment { border-left: 3px solid #007bff; padding-left: 15px; margin: 10px 0; }
        .reply { border-left: 3px solid #28a745; margin-left: 20px; padding-left: 15px; }
        .comment-meta { color: #666; font-size: 12px; }
        .stats-badge { background: #f8f9fa; padding: 4px 8px; border-radius: 4px; margin-right: 10px; }
    </style>
</head>
<body>
    <h1>用户交互模拟报告</h1>
    <p>会话ID: {session_id}</p>
    
    <div class="post">
        <div class="post-header">{post_title}</div>
        <div class="post-content">{post_content}</div>
        <div class="post-stats">
            <span class="stats-badge">👍 {likes_count} 赞</span>
            <span class="stats-badge">💬 {comments_count} 评论</span>
            <span class="stats-badge">📤 {shares_count} 分享</span>
        </div>
        
        <div class="comments-section">
            {comments_html}
        </div>
    </div>
</body>
</html>
        """

        # 生成评论HTML
        comments_html = ""
        for comment in display_data.get("comments", []):
            comments_html += f"""
            <div class="comment">
                <div>{comment['content']}</div>
                <div class="comment-meta">
                    用户: {comment['user_id']} | 
                    时间: {comment['timestamp']} | 
                    👍 {comment['likes_count']} 赞
                </div>
            """

            # 添加回复
            for reply in comment.get("replies", []):
                comments_html += f"""
                <div class="reply">
                    <div>{reply['content']}</div>
                    <div class="comment-meta">
                        用户: {reply['user_id']} | 
                        时间: {reply['timestamp']} | 
                        👍 {reply['likes_count']} 赞
                    </div>
                </div>
                """

            comments_html += "</div>"

        return html_template.format(
            session_id=display_data.get("session_id", "unknown"),
            post_title=display_data.get("post", {}).get("title", ""),
            post_content=display_data.get("post", {}).get("content", ""),
            likes_count=display_data.get("stats", {}).get("likes", 0),
            comments_count=display_data.get("stats", {}).get("comments", 0),
            shares_count=display_data.get("stats", {}).get("shares", 0),
            comments_html=comments_html
        )
