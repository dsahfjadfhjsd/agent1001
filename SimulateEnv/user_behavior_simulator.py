# -*- coding: utf-8 -*-
"""
用户行为模拟器

基于用户画像生成相应的社交媒体行为
支持并发调用AI模型来生成用户行为
"""

import asyncio
import json
import random
import os
import os.path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

try:
    # 尝试相对导入
    from .interaction_core import UserAction, ActionType
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from interaction_core import UserAction, ActionType


@dataclass
class UserThinkingResult:
    """用户思考结果"""
    user_id: str
    action: Optional[UserAction]  # 用户行为
    thinking_process: str  # 思考过程
    stance_before: float   # 行为前立场值
    stance_after: float    # 行为后立场值
    sentiment_before: float  # 行为前情感值
    sentiment_after: float   # 行为后情感值
    content_seen: List[str]  # 看到的内容


@dataclass
class SimulationConfig:
    """模拟配置"""
    max_concurrent_requests: int = 5
    request_timeout: int = 60
    model_name: str = "qwen-max"
    # 备用方案参数
    action_probability: float = 1.0  # 用户采取行动的概率
    # 新增：Prompt导出配置
    export_prompts: bool = False  # 是否导出所有prompts到文件
    prompt_export_dir: str = None  # prompt导出目录
    # 新增：多模态分析配置
    enable_multimodal: bool = False  # 是否启用多模态分析
    multimodal_model: str = "qwen-vl-max"  # 多模态模型名称
    multimodal_max_images: int = 5  # 每个帖子最多处理的图片数量
    multimodal_timeout: int = 60  # 多模态分析超时时间（秒）
    multimodal_fallback_on_error: bool = True  # URL失效或分析失败时是否回退到原始内容
    # 新增：多模态缓存配置
    multimodal_use_cache: bool = True  # 是否使用多模态分析缓存
    multimodal_cache_dir: str = None  # 多模态缓存目录路径
    multimodal_cache_filename: str = None  # 多模态缓存文件名（不含扩展名），为None时使用默认名称


class UserBehaviorSimulator:
    """用户行为模拟器"""

    def __init__(self, config: SimulationConfig = None):
        """
        初始化模拟器

        Args:
            config: 模拟配置
        """
        # 加载环境变量
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config', '.env'))

        self.config = config or SimulationConfig()

        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )

        # 创建信号量控制并发
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # 初始化prompt导出功能
        if self.config.export_prompts:
            self._init_prompt_export()

    async def close(self):
        """关闭客户端连接"""
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
        except:
            pass

    def _select_comments_for_display(self, comments):
        """
        智能选择要展示的评论：
        - 按点赞数和回复数排序的前3个热门评论
        - 随机选择2个其他评论
        - 总共返回最多5个评论
        """
        if not comments:
            return []

        # 计算每个评论的热度分数（点赞数 + 回复数 * 2）
        def get_heat_score(comment):
            likes = comment.get('likes', 0)
            reply_count = len(comment.get('sub_comments', []))
            return likes + reply_count * 2

        # 按热度排序，获取前3个热门评论
        sorted_comments = sorted(comments, key=get_heat_score, reverse=True)
        hot_comments = sorted_comments[:3]

        # 如果评论总数不足5个，直接返回所有评论
        if len(comments) <= 5:
            return comments[:5]

        # 从剩余评论中随机选择2个
        remaining_comments = [c for c in comments if c not in hot_comments]
        if len(remaining_comments) >= 2:
            import random
            random_comments = random.sample(remaining_comments, 2)
        else:
            random_comments = remaining_comments

        # 合并并返回（保持原始顺序）
        selected_ids = {c['comment_id'] for c in hot_comments + random_comments}
        result = [c for c in comments if c['comment_id'] in selected_ids]

        return result[:5]  # 确保最多5个

    def _init_prompt_export(self):
        """初始化prompt导出功能"""
        from datetime import datetime
        from pathlib import Path

        # 创建导出目录
        export_dir = Path(self.config.prompt_export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # 创建本次会话的导出文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prompt_export_file = export_dir / f"prompts_{timestamp}.txt"

        # 计数器
        self.prompt_counter = 0

        # 写入文件头
        with open(self.prompt_export_file, 'w', encoding='utf-8') as f:
            f.write(f"=== AI Prompt Export File ===\n")
            f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.config.model_name}\n")
            f.write("=" * 80 + "\n\n")

        print(f"📝 Prompt export enabled, saved to: {self.prompt_export_file}")

    def _export_prompt(self, user_id: str, prompt: str, ai_response: str = None):
        """导出单个prompt到文件"""
        if not self.config.export_prompts or not hasattr(self, 'prompt_export_file'):
            return

        self.prompt_counter += 1

        try:
            with open(self.prompt_export_file, 'a', encoding='utf-8') as f:
                f.write(f"📝 Prompt #{self.prompt_counter}\n")
                f.write(f"👤 User ID: {user_id}\n")
                f.write(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
                f.write("🤖 Input Prompt:\n")
                f.write(prompt)
                f.write("\n" + "-" * 50 + "\n")

                if ai_response:
                    f.write("🎯 AI Response:\n")
                    f.write(ai_response)
                    f.write("\n" + "-" * 50 + "\n")

                f.write("\n" + "=" * 80 + "\n\n")

        except Exception as e:
            print(f"Failed to export prompt: {e}")

    def get_prompt_export_path(self) -> str:
        """获取prompt导出文件路径"""
        if hasattr(self, 'prompt_export_file'):
            return str(self.prompt_export_file)
        return ""

    async def simulate_user_behavior_with_thinking(
        self,
        user_profile: Dict[str, Any],
        environment_state: Dict[str, Any],
        round_number: int = 1,
        user_memory: List[Dict] = None
    ) -> Optional[UserThinkingResult]:
        """
        模拟单个用户的行为，包含思考过程和认知变化

        Args:
            user_profile: 用户画像
            environment_state: 当前环境状态
            round_number: 当前轮次
            user_memory: 用户历史记忆

        Returns:
            用户思考结果，如果用户不采取行动则返回None
        """
        # 决定用户是否采取行动
        if random.random() > self.config.action_probability:
            return None

        # 使用AI决定具体行为并获取思考过程
        async with self.semaphore:
            try:
                thinking_result = await self._get_ai_action_with_thinking(
                    user_profile, environment_state, user_memory
                )
                if thinking_result:
                    # 创建用户行为
                    if thinking_result.get('action_type') and thinking_result.get('action_type') != 'no_action':
                        action = self._create_user_action(
                            user_profile['user_id'],
                            thinking_result,
                            round_number,
                            environment_state  # 传递环境状态用于ID验证
                        )
                    else:
                        action = None

                    # 创建完整的思考结果
                    result = UserThinkingResult(
                        user_id=user_profile['user_id'],
                        action=action,
                        thinking_process=thinking_result.get('thinking_process', ''),
                        stance_before=float(user_profile.get('stance_value', 0.0)),
                        stance_after=float(thinking_result.get('stance_after', user_profile.get('stance_value', 0.0))),
                        sentiment_before=float(user_profile.get('sentiment_value', 0.0)),
                        sentiment_after=float(thinking_result.get('sentiment_after', user_profile.get('sentiment_value', 0.0))),
                        content_seen=self._extract_content_seen(environment_state)
                    )
                    return result

            except Exception as e:
                print(f"User {user_profile['user_id']} behavior simulation failed: {e}")
                return None

        return None

    async def _get_ai_action_with_thinking(
        self,
        user_profile: Dict[str, Any],
        environment_state: Dict[str, Any],
        user_memory: List[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        使用AI决定用户行为并获取思考过程

        Args:
            user_profile: 用户画像
            environment_state: 环境状态
            user_memory: 用户历史记忆

        Returns:
            包含行为决策、思考过程和认知变化的字典
        """
        # 构建增强的提示词
        prompt = self._build_thinking_prompt(user_profile, environment_state, user_memory)

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a social media user behavior simulator. Based on user profiles, current environment, and historical memory, simulate the user's thinking process and decide on actions.

User behavior decision rules:
- Activity level: Lower activity levels make users less likely to take action
- Behavior preference: Prioritize liking > commenting on posts >= replying to comments
- When comments exist, prioritize interaction with other users (reply to comments or like comments)
- Comment content should not exceed 30 words, fit social media context, reflect personal opinions, avoid repeating others' content
- Comment language should match the post language and user context (English for English posts, Chinese for Chinese posts, etc.)

Available action types:
- like_post: Like a post, use post ID as target_id
- comment_post: Comment on a post, use post ID as target_id, provide action_content
- like_comment: Like a comment, use comment ID as target_id
- comment_comment: Reply to a comment, use comment ID as target_id, provide action_content
- no_action: Take no action

Important reminders:
Even if you don't intend to take action, please output no_action along with your thinking process to maintain format consistency.
When comments exist, consider comment_comment (reply to comment) or like_comment (like comment) to promote user interaction.

Please respond strictly in JSON format, including:
- thinking_process: Detailed thinking process (within 100 words)
- action_type: Action type (like_post/comment_post/like_comment/comment_comment/no_action)
- action_content: Comment content (only required when action_type is comment_post or comment_comment)
- target_id: Target ID (use post ID for liking/commenting posts; use comment ID for liking/replying comments; must use real IDs)
- stance_after: Stance value after viewing content (-1 to 1 numeric value)
- sentiment_after: Sentiment value after viewing content (-1 to 1 numeric value)"""
                        },
                        {"role": "user", "content": prompt}
                    ],
                ),
                timeout=self.config.request_timeout
            )

            content = response.choices[0].message.content.strip()

            # 导出prompt（如果启用）
            if self.config.export_prompts:
                user_id = user_profile.get('user_id', 'unknown')
                self._export_prompt(user_id, prompt, content)

            # 解析JSON响应
            try:
                decision = json.loads(content)
                return decision
            except json.JSONDecodeError:
                # 尝试提取JSON部分
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    return decision
                else:
                    print(f"Unable to parse AI response: {content}")
                    return None

        except asyncio.TimeoutError:
            print("AI request timeout")
            return None
        except Exception as e:
            print(f"AI request failed: {e}")
            return None

    def _build_thinking_prompt(
        self,
        user_profile: Dict[str, Any],
        environment_state: Dict[str, Any],
        user_memory: List[Dict] = None
    ) -> str:
        """
        构建包含思考过程的增强提示词

        Args:
            user_profile: 用户画像
            environment_state: 环境状态
            user_memory: 用户历史记忆

        Returns:
            增强的提示词字符串
        """
        # 提取用户特征
        age_group = user_profile.get('age_group', '未知')
        gender = user_profile.get('gender', '未知')
        occupation = user_profile.get('occupation', '未知')
        activity_level = user_profile.get('activity_level', '中等')
        stance = user_profile.get('stance', '中立')
        stance_keywords = user_profile.get('stance_keywords', '无')
        sentiment = user_profile.get('sentiment', '中立')
        sentiment_keywords = user_profile.get('sentiment_keywords', '无')
        stance_value = user_profile.get('stance_value', 0.0)
        sentiment_value = user_profile.get('sentiment_value', 0.0)
        intent_keywords = user_profile.get('intent_keywords', '无')

        # 提取环境信息
        post = environment_state['post']
        comments = environment_state['primary_comments']

        prompt = f"""
User Profile:
- Age Group: {age_group}
- Gender: {gender}
- Occupation: {occupation}
- Activity Level: {activity_level}
- Stance: {stance}:{stance_keywords} (Current value: {stance_value}, Support=1, Oppose=-1, Neutral=0)
- Sentiment Tendency: {sentiment}:{sentiment_keywords} (Current value: {sentiment_value}, Positive=1, Negative=-1, Neutral=0)
- Intent Keywords: {intent_keywords}

Current Environment:
Post Content: "{post['content']}"
Post ID: {post['post_id']}
Post Likes: {post['likes']}

Existing Comments:
"""

        if comments:
            # 选择评论：按点赞数和回复数排序的前3个 + 随机2个
            selected_comments = self._select_comments_for_display(comments)

            for i, comment in enumerate(selected_comments, 1):
                sub_comment_info = ""
                if comment.get('sub_comments'):
                    sub_count = len(comment['sub_comments'])
                    if sub_count > 0:
                        sub_comment_info = f" ({sub_count} replies)"

                prompt += f"{i}. Comment ID: {comment['comment_id']} [User {comment['author_id'][-8:]}]: {comment['content']} "
                prompt += f"[Likes: {comment['likes']}]{sub_comment_info}\n"

                # 显示该评论的前2条二级评论
                if comment.get('sub_comments'):
                    for j, sub_comment in enumerate(comment['sub_comments'][:2], 1):
                        prompt += f"    └─ Reply {j}: [User {sub_comment['author_id'][-8:]}] {sub_comment['content']} "
                        prompt += f"[Likes: {sub_comment.get('likes', 0)}]\n"
        else:
            prompt += "No comments yet\n"

        # 添加历史记忆
        if user_memory:
            user_memory = user_memory[-5:]  # 最多只保留最近5次交互
            prompt += f"\nUser Historical Memory (Recent {len(user_memory)} interactions):\n"
            for i, memory in enumerate(user_memory, 1):
                prompt += f"{i}. Round {memory.get('round_number', '?')}: {memory.get('thinking_process', 'No record')[:60]}...\n"
                prompt += f"   Action: {memory.get('action_taken', 'None')} | Stance change: {memory.get('stance_before', 0):.1f}→{memory.get('stance_after', 0):.1f} | Sentiment change: {memory.get('sentiment_before', 0):.1f}→{memory.get('sentiment_after', 0):.1f}\n"

        prompt += f"""

Based on the user profile and current environment, assume you are this user and decide whether to take action and what action to take.

Thinking Steps:
1. Carefully read the post content and existing comments
2. Think based on your user profile and historical memory
3. Consider possible changes in stance and sentiment after viewing this content
4. Decide whether to take action and what action to take
5. When outputting, pay attention to language choice (Chinese/English/other) that matches the user profile and current context
"""

        return prompt

    def _extract_content_seen(self, environment_state: Dict[str, Any]) -> List[str]:
        """
        提取用户看到的内容列表

        Args:
            environment_state: 环境状态

        Returns:
            内容列表
        """
        content_seen = []

        # 添加帖子内容
        post = environment_state['post']
        content_seen.append(f"Post: {post['content']}")

        # 添加评论内容
        comments = environment_state['primary_comments']
        # 使用智能评论选择
        selected_comments = self._select_comments_for_display(comments)

        for comment in selected_comments:  # 使用智能选择的评论
            content_seen.append(f"Comment: {comment['content']}")

            # 添加子评论
            if comment.get('sub_comments'):
                for sub_comment in comment['sub_comments'][:2]:  # 每个评论最多看2条回复
                    content_seen.append(f"Reply: {sub_comment['content']}")

        return content_seen

    def _create_user_action(
        self,
        user_id: str,
        action_decision: Dict[str, Any],
        round_number: int,
        environment_state: Dict[str, Any] = None
    ) -> Optional[UserAction]:
        """
        根据AI决策创建用户行为

        Args:
            user_id: 用户ID
            action_decision: AI决策结果
            round_number: 轮次号

        Returns:
            用户行为实例
        """
        # 尝试不同的键名来获取行为类型
        action_type_str = (action_decision.get('action') or
                           action_decision.get('action_type') or
                           action_decision.get('type'))

        if action_type_str == 'no_action' or action_type_str is None:
            return None

        # 映射行为类型
        action_type_map = {
            'like_post': ActionType.LIKE_POST,
            'comment_post': ActionType.COMMENT_POST,
            'like_comment': ActionType.LIKE_COMMENT,
            'comment_comment': ActionType.COMMENT_COMMENT
        }

        if action_type_str not in action_type_map:
            print(f"Unknown action type: {action_type_str}")
            return None

        action_type = action_type_map[action_type_str]
        target_id = action_decision.get('target_id', '')
        content = action_decision.get('content', '') or action_decision.get('action_content', '')

        # 验证和修正target_id
        if environment_state:
            post_id = environment_state['post']['post_id']
            comments = environment_state.get('primary_comments', [])
            comment_ids = [comment['comment_id'] for comment in comments]

            # 如果是对帖子的操作
            if action_type_str in ['like_post', 'comment_post']:
                if target_id != post_id:
                    print(f"Auto-correcting post ID: {target_id} → {post_id}")
                    target_id = post_id

            # 如果是对评论的操作但ID不匹配，使用第一个评论的ID
            elif action_type_str in ['like_comment', 'comment_comment']:
                if target_id not in comment_ids and comment_ids:
                    print(f"Auto-correcting comment ID: {target_id} → {comment_ids[0]}")
                    target_id = comment_ids[0]

        if not target_id:
            print(f"Missing target ID: {action_decision}")
            return None

        return UserAction(
            action_id="",  # 会自动生成
            user_id=user_id,
            action_type=action_type,
            target_id=target_id,
            content=content if content else None,
            round_number=round_number
        )
