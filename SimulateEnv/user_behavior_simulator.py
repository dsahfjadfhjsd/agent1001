"""
用户行为模拟器

基于用户画像生成相应的社交媒体行为
支持并发调用AI模型来生成用户行为
"""

import asyncio
import json
import random
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import openai
import sys
from openai import AsyncOpenAI
from dotenv import load_dotenv

try:
    # 尝试相对导入
    from .interaction_core import InteractionEnvironment, UserAction, ActionType
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from interaction_core import InteractionEnvironment, UserAction, ActionType


@dataclass
class SimulationConfig:
    """模拟配置"""
    max_concurrent_requests: int = 5
    request_timeout: int = 60
    model_name: str = "qwen-flash"
    max_tokens: int = 500
    temperature: float = 0.7
    action_probability: float = 0.7  # 用户采取行动的概率
    comment_probability: float = 0.3  # 在决定行动时选择评论而非点赞的概率


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

    async def close(self):
        """关闭客户端连接"""
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
        except:
            pass

    async def simulate_user_behavior(
        self,
        user_profile: Dict[str, Any],
        environment_state: Dict[str, Any],
        round_number: int = 1
    ) -> Optional[UserAction]:
        """
        模拟单个用户的行为

        Args:
            user_profile: 用户画像
            environment_state: 当前环境状态
            round_number: 当前轮次

        Returns:
            用户行为，如果用户不采取行动则返回None
        """
        # 决定用户是否采取行动
        if random.random() > self.config.action_probability:
            return None

        # 使用AI决定具体行为
        async with self.semaphore:
            try:
                action_decision = await self._get_ai_action_decision(user_profile, environment_state)
                if action_decision:
                    action = self._create_user_action(
                        user_profile['user_id'],
                        action_decision,
                        round_number
                    )
                    return action
            except Exception as e:
                print(f"用户 {user_profile['user_id']} 行为模拟失败: {e}")
                return None

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

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤出成功的行为
        actions = []
        for result in results:
            if isinstance(result, UserAction):
                actions.append(result)
            elif isinstance(result, Exception):
                print(f"用户行为模拟异常: {result}")

        return actions

    async def _get_ai_action_decision(
        self,
        user_profile: Dict[str, Any],
        environment_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        使用AI决定用户行为

        Args:
            user_profile: 用户画像
            environment_state: 环境状态

        Returns:
            行为决策字典
        """
        # 构建提示词
        prompt = self._build_behavior_prompt(user_profile, environment_state)

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个社交媒体用户行为模拟器。根据用户画像和当前环境，决定用户的行为。请严格按照JSON格式回复。"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    # max_tokens=self.config.max_tokens,
                    # temperature=self.config.temperature
                ),
                timeout=self.config.request_timeout
            )

            content = response.choices[0].message.content.strip()

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

    def _build_behavior_prompt(
        self,
        user_profile: Dict[str, Any],
        environment_state: Dict[str, Any]
    ) -> str:
        """
        构建行为决策提示词

        Args:
            user_profile: 用户画像
            environment_state: 环境状态

        Returns:
            提示词字符串
        """
        # 提取用户特征
        age_group = user_profile.get('age_group', '未知')
        gender = user_profile.get('gender', '未知')
        occupation = user_profile.get('occupation', '未知')
        activity_level = user_profile.get('activity_level', '中等')
        stance = user_profile.get('stance', '中立')
        sentiment = user_profile.get('sentiment', '中立')

        # 提取环境信息
        post = environment_state['post']
        comments = environment_state['primary_comments']

        prompt = f"""
用户画像：
- 年龄组：{age_group}
- 性别：{gender}
- 职业：{occupation}
- 活跃度：{activity_level}
- 立场：{stance}
- 情感倾向：{sentiment}

当前环境：
帖子内容："{post['content']}"
帖子点赞数：{post['likes']}

已有评论：
"""

        if comments:
            for i, comment in enumerate(comments[:5], 1):  # 只显示前5条评论
                prompt += f"{i}. {comment['content']} (点赞: {comment['likes']})\n"

                # 显示部分二级评论
                if comment.get('sub_comments'):
                    for sub in comment['sub_comments'][:2]:  # 只显示前2条二级评论
                        prompt += f"   └ {sub['content']} (点赞: {sub['likes']})\n"
        else:
            prompt += "暂无评论\n"

        prompt += f"""
请根据用户画像和当前环境，假设你是该用户，判断是否采取行动以及采取什么行动。
通常根据用户活跃度越低，越可能不采取行动
而若是行动，行为倾向高低一般为：点赞 > 评论 = 回复

可选行为：
1. like_post - 点赞帖子
2. comment_post - 评论帖子
3. like_comment - 点赞某条评论（需要指定comment_id）
4. comment_comment - 回复某条评论（需要指定comment_id和回复内容）

请回复JSON格式：
{{
    "action": "行为类型",
    "target_id": "目标ID（帖子ID或评论ID）",
    "content": "评论内容（如果是评论行为）",
    "reasoning": "行为理由"
}}

如果决定不采取行动，请回复：
{{
    "action": "no_action",
    "reasoning": "不行动的理由"
}}

可用的目标ID：
- 帖子ID: {post['post_id']}
"""

        if comments:
            prompt += "- 评论ID:\n"
            for comment in comments:
                prompt += f"  * {comment['comment_id']}\n"
                for sub in comment.get('sub_comments', []):
                    prompt += f"    - {sub['comment_id']}\n"

        # 将prompt持续输出到一个txt文件中
        with open("user_behavior_prompt.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n=== 时间戳: {datetime.now().strftime('%H:%M:%S')} ===\n")
            f.write(prompt)
            f.write("\n" + "="*50 + "\n")

        return prompt

    def _create_user_action(
        self,
        user_id: str,
        action_decision: Dict[str, Any],
        round_number: int
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
        action_type_str = action_decision.get('action')

        if action_type_str == 'no_action':
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
        content = action_decision.get('content', '')

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
            if comments and random.random() < 0.3:  # 30%概率回复评论
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


if __name__ == "__main__":
    # 测试代码
    async def test_simulator():
        import sys
        import os

        # 添加项目根目录到路径
        project_root = os.path.dirname(os.path.dirname(__file__))
        sys.path.append(project_root)

        from UserAgent.user_profile_manager import UserProfileManager
        from SimulateEnv.interaction_core import InteractionEnvironment

        # 创建测试环境
        env = InteractionEnvironment("人工智能的发展对社会有什么影响？")

        # 加载用户画像
        manager = UserProfileManager()
        manager.load_users_from_file("test_10.csv")  # 假设存在这个文件
        users = manager.get_all_users()[:3]  # 使用前3个用户

        # 创建模拟器
        simulator = UserBehaviorSimulator()

        # 模拟用户行为
        env_state = env.get_environment_state()
        actions = await simulator.simulate_multiple_users(users, env_state, 1)

        print(f"生成了 {len(actions)} 个用户行为:")
        for action in actions:
            print(f"- 用户 {action.user_id}: {action.action_type.value} -> {action.target_id}")
            if action.content:
                print(f"  内容: {action.content}")

        # 关闭客户端连接
        await simulator.close()

    # 运行测试 - 使用更安全的方式
    import os
    import sys

    # 添加当前目录到路径以导入 async_utils
    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)

    from async_utils import run_async_simple
    run_async_simple(test_simulator())
