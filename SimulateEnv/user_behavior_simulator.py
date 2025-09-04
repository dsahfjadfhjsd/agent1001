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
    max_tokens: int = 500
    temperature: float = 0.7
    action_probability: float = 0.7  # 用户采取行动的概率
    comment_probability: float = 0.3  # 在决定行动时选择评论而非点赞的概率
    # 新增：Prompt导出配置
    export_prompts: bool = False  # 是否导出所有prompts到文件
    prompt_export_dir: str = "Output/prompt_exports"  # prompt导出目录


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
            f.write(f"=== AI提示词导出文件 ===\n")
            f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {self.config.model_name}\n")
            f.write("=" * 80 + "\n\n")

        print(f"📝 Prompt导出已启用，保存到: {self.prompt_export_file}")

    def _export_prompt(self, user_id: str, prompt: str, ai_response: str = None):
        """导出单个prompt到文件"""
        if not self.config.export_prompts or not hasattr(self, 'prompt_export_file'):
            return

        self.prompt_counter += 1

        try:
            with open(self.prompt_export_file, 'a', encoding='utf-8') as f:
                f.write(f"📝 Prompt #{self.prompt_counter}\n")
                f.write(f"👤 用户ID: {user_id}\n")
                f.write(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
                f.write("🤖 输入提示词:\n")
                f.write(prompt)
                f.write("\n" + "-" * 50 + "\n")

                if ai_response:
                    f.write("🎯 AI响应:\n")
                    f.write(ai_response)
                    f.write("\n" + "-" * 50 + "\n")

                f.write("\n" + "=" * 80 + "\n\n")

        except Exception as e:
            print(f"导出prompt失败: {e}")

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
                print(f"用户 {user_profile['user_id']} 行为模拟失败: {e}")
                return None

        return None

    async def simulate_user_behavior(
        self,
        user_profile: Dict[str, Any],
        environment_state: Dict[str, Any],
        round_number: int = 1
    ) -> Optional[UserAction]:
        """
        模拟单个用户的行为（保持向后兼容性）

        Args:
            user_profile: 用户画像
            environment_state: 当前环境状态
            round_number: 当前轮次

        Returns:
            用户行为，如果用户不采取行动则返回None
        """
        thinking_result = await self.simulate_user_behavior_with_thinking(
            user_profile, environment_state, round_number
        )

        if thinking_result and thinking_result.action:
            return thinking_result.action
        return None

    async def simulate_multiple_users(
        self,
        user_profiles: List[Dict[str, Any]],
        environment_state: Dict[str, Any],
        round_number: int = 1
    ) -> List[UserAction]:
        """
        并发模拟多个用户的行为

        Args:
            user_profiles: 用户画像列表
            environment_state: 当前环境状态
            round_number: 当前轮次

        Returns:
            用户行为列表
        """
        tasks = []
        for user_profile in user_profiles:
            task = self.simulate_user_behavior(user_profile, environment_state, round_number)
            tasks.append(task)

        # 使用tqdm显示进度条
        print(f"🤖 正在模拟 {len(user_profiles)} 个用户的行为...")

        # 包装任务以处理异常
        async def safe_task(task):
            try:
                return await task
            except Exception as e:
                return e

        safe_tasks = [safe_task(task) for task in tasks]
        results = await tqdm.gather(
            *safe_tasks,
            desc=f"第{round_number}轮API调用",
            unit="用户",
            colour="green"
        )

        # 过滤出成功的行为
        actions = []
        successful_count = 0
        error_count = 0

        for result in results:
            if isinstance(result, UserAction):
                actions.append(result)
                successful_count += 1
            elif isinstance(result, Exception):
                error_count += 1
                print(f"⚠️  用户行为模拟异常: {result}")
            else:
                # None result (用户未采取行动)
                pass

        print(f"✅ 完成模拟 - 成功: {successful_count}, 错误: {error_count}, 无行动: {len(user_profiles) - successful_count - error_count}")
        return actions

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
                            "content": """你是一个社交媒体用户行为模拟器。根据用户画像、当前环境和历史记忆，模拟用户的思考过程并决定行为。

用户行为决策规则：
- 根据活跃度：活跃度越低，越可能不采取行动
- 行为偏好：优先点赞 > 评论帖子 >= 回复评论
- 当已有评论时，优先考虑与其他用户互动（回复评论或点赞评论）
- 评论内容不超过30字，要符合网络用户语境，体现个人观点，避免重复他人内容

可选的行为类型说明：
- like_post: 点赞帖子，target_id使用帖子ID
- comment_post: 评论帖子，target_id使用帖子ID，需要提供action_content
- like_comment: 点赞评论，target_id使用评论ID
- comment_comment: 回复评论，target_id使用要回复的评论ID，需要提供action_content
- no_action: 不采取任何行动

特别提醒：
即使你不打算采取行动，也请输出no_action，以及你的思考过程等，保持格式完整。
当存在评论时，可以考虑comment_comment（回复评论）或like_comment（点赞评论），这能促进用户间的互动交流。

请严格按照JSON格式回复，包含：
- thinking_process: 详细的思考过程（100字以内）
- action_type: 行为类型（like_post/comment_post/like_comment/comment_comment/no_action）
- action_content: 评论内容（仅当action_type为comment_post或comment_comment时需要）
- target_id: 目标ID（如果是点赞/评论帖子，使用帖子ID；如果是点赞/回复评论，使用评论ID；必须使用真实ID）
- stance_after: 看完内容后的立场值（-1到1的数值）
- sentiment_after: 看完内容后的情感值（-1到1的数值）"""
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
                    print(f"无法解析AI响应: {content}")
                    return None

        except asyncio.TimeoutError:
            print("AI请求超时")
            return None
        except Exception as e:
            print(f"AI请求失败: {e}")
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
        sentiment = user_profile.get('sentiment', '中立')
        stance_value = user_profile.get('stance_value', 0.0)
        sentiment_value = user_profile.get('sentiment_value', 0.0)

        # 提取环境信息
        post = environment_state['post']
        comments = environment_state['primary_comments']

        prompt = f"""
用户画像：
- 年龄组：{age_group}
- 性别：{gender}
- 职业：{occupation}
- 活跃度：{activity_level}
- 立场：{stance}（当前值：{stance_value}，-1完全批判小米，1完全支持小米，0中立）
- 情感倾向：{sentiment}（当前值：{sentiment_value}，-1消极，1积极，0中立）

当前环境：
帖子内容："{post['content']}"
帖子ID：{post['post_id']}
帖子点赞数：{post['likes']}

已有评论：
"""

        if comments:
            # 选择评论：按点赞数和回复数排序的前3个 + 随机2个
            selected_comments = self._select_comments_for_display(comments)

            for i, comment in enumerate(selected_comments, 1):
                sub_comment_info = ""
                if comment.get('sub_comments'):
                    sub_count = len(comment['sub_comments'])
                    if sub_count > 0:
                        sub_comment_info = f"（有{sub_count}条回复）"

                prompt += f"{i}. 评论ID：{comment['comment_id']} 【用户{comment['author_id'][-8:]}】: {comment['content']} "
                prompt += f"[点赞数: {comment['likes']}] {sub_comment_info}\n"

                # 显示该评论的前2条二级评论
                if comment.get('sub_comments'):
                    for j, sub_comment in enumerate(comment['sub_comments'][:2], 1):
                        prompt += f"    └─ 回复{j}：【用户{sub_comment['author_id'][-8:]}】{sub_comment['content']} "
                        prompt += f"[点赞数: {sub_comment.get('likes', 0)}]\n"
        else:
            prompt += "暂无评论\n"

        # 添加历史记忆
        if user_memory:
            user_memory = user_memory[-5:]  # 最多只保留最近5次交互
            prompt += f"\n用户历史记忆（最近{len(user_memory)}次交互）：\n"
            for i, memory in enumerate(user_memory, 1):
                prompt += f"{i}. 轮次{memory.get('round_number', '?')}：{memory.get('thinking_process', '无记录')[:60]}...\n"
                prompt += f"   行为：{memory.get('action_taken', '无')} | 立场变化：{memory.get('stance_before', 0):.1f}→{memory.get('stance_after', 0):.1f} | 情感变化：{memory.get('sentiment_before', 0):.1f}→{memory.get('sentiment_after', 0):.1f}\n"

        prompt += f"""

请根据用户画像和当前环境，假设你是该用户，判断是否采取行动以及采取什么行动。

思考步骤：
1. 仔细阅读帖子内容和已有评论
2. 结合你的用户画像和历史记忆进行思考
3. 考虑看到这些内容后立场和情感的可能变化
4. 决定是否要采取行动以及采取什么行动
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
        content_seen.append(f"帖子：{post['content']}")

        # 添加评论内容
        comments = environment_state['primary_comments']
        # 使用智能评论选择
        selected_comments = self._select_comments_for_display(comments)

        for comment in selected_comments:  # 使用智能选择的评论
            content_seen.append(f"评论：{comment['content']}")

            # 添加子评论
            if comment.get('sub_comments'):
                for sub_comment in comment['sub_comments'][:2]:  # 每个评论最多看2条回复
                    content_seen.append(f"回复：{sub_comment['content']}")

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
            print(f"未知的行为类型: {action_type_str}")
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
                    print(f"自动修正帖子ID: {target_id} → {post_id}")
                    target_id = post_id

            # 如果是对评论的操作但ID不匹配，使用第一个评论的ID
            elif action_type_str in ['like_comment', 'comment_comment']:
                if target_id not in comment_ids and comment_ids:
                    print(f"自动修正评论ID: {target_id} → {comment_ids[0]}")
                    target_id = comment_ids[0]

        if not target_id:
            print(f"缺少目标ID: {action_decision}")
            return None

        return UserAction(
            action_id="",  # 会自动生成
            user_id=user_id,
            action_type=action_type,
            target_id=target_id,
            content=content if content else None,
            round_number=round_number
        )

    def generate_fallback_action(
        self,
        user_id: str,
        environment_state: Dict[str, Any],
        round_number: int
    ) -> Optional[UserAction]:
        """
        生成备用行为（当AI调用失败时使用）

        Args:
            user_id: 用户ID
            environment_state: 环境状态
            round_number: 轮次号

        Returns:
            备用用户行为
        """
        post = environment_state['post']
        comments = environment_state['primary_comments']

        # 随机选择行为类型
        if random.random() < self.config.comment_probability:
            # 评论行为
            if comments and random.random() < 0.5:  # 50%概率回复评论（提高概率）
                comment = random.choice(comments)
                return UserAction(
                    action_id="",
                    user_id=user_id,
                    action_type=ActionType.COMMENT_COMMENT,
                    target_id=comment['comment_id'],
                    content="同意你的观点！",
                    round_number=round_number
                )
            else:  # 评论帖子
                fallback_comments = [
                    "很有意思的内容",
                    "学到了",
                    "赞同",
                    "有道理",
                    "支持"
                ]
                return UserAction(
                    action_id="",
                    user_id=user_id,
                    action_type=ActionType.COMMENT_POST,
                    target_id=post['post_id'],
                    content=random.choice(fallback_comments),
                    round_number=round_number
                )
        else:
            # 点赞行为
            if comments and random.random() < 0.4:  # 40%概率点赞评论
                comment = random.choice(comments)
                return UserAction(
                    action_id="",
                    user_id=user_id,
                    action_type=ActionType.LIKE_COMMENT,
                    target_id=comment['comment_id'],
                    round_number=round_number
                )
            else:  # 点赞帖子
                return UserAction(
                    action_id="",
                    user_id=user_id,
                    action_type=ActionType.LIKE_POST,
                    target_id=post['post_id'],
                    round_number=round_number
                )
