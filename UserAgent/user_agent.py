"""
用户智能体核心实现
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import concurrent.futures
from functools import partial
import openai
from openai import OpenAI

from .user_profile import UserProfile, UserMemory
from Config.settings import config


class UserAction:
    """用户动作"""

    def __init__(self, action_type: str, target_id: str, content: str = "", timestamp: datetime = None):
        self.action_type = action_type  # like, comment, share
        self.target_id = target_id  # 目标内容ID
        self.content = content  # 动作内容（如评论文本）
        self.timestamp = timestamp or datetime.now()
        self.action_id = f"{action_type}_{target_id}_{self.timestamp.timestamp()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "target_id": self.target_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


class UserAgent:
    """用户智能体"""

    def __init__(self, profile: UserProfile, memory: UserMemory = None):
        self.profile = profile
        self.memory = memory or UserMemory(user_id=profile.user_id)
        self.client = OpenAI(
            api_key=config.model.api_key,
            base_url=config.model.base_url
        )

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return f"""
你是一个用户模拟智能体，需要根据给定的用户画像来模拟真实用户的行为。

用户画像：
- 用户ID: {self.profile.user_id}
- 年龄: {self.profile.age}
- 性别: {self.profile.gender.value}
- 职业: {self.profile.occupation}
- 教育水平: {self.profile.education_level}
- 地理位置: {self.profile.location}
- 对当前事件的立场: {self.profile.stance.value}
- 对当前事件的情感: {self.profile.emotion.value}
- 对当前事件的意图: {self.profile.intent.value}
- 活跃度: {self.profile.activity_level}
- 社会影响力: {self.profile.social_influence}

你需要根据这个画像来决定用户在看到内容时的行为。
可能的动作包括：like（点赞）、comment（评论）、share（分享）、ignore（忽略）。

请始终保持角色一致性，回复应该符合用户的画像特征。
回复格式必须是JSON格式，包含以下字段：
{{
    "action": "动作类型（like/comment/share/ignore）",
    "content": "如果是评论，填写评论内容；其他动作可以为空",
    "reasoning": "选择这个动作的原因"
}}
"""

    def _build_user_prompt(self, environment_content: Dict[str, Any]) -> str:
        """构建用户提示词"""
        # 获取最近的记忆
        recent_interactions = self.memory.get_recent_interactions(3)

        prompt = f"""
当前环境内容：
{json.dumps(environment_content, ensure_ascii=False, indent=2)}

你的最近交互记录：
{json.dumps(recent_interactions, ensure_ascii=False, indent=2) if recent_interactions else "无"}

请根据你的用户画像和当前看到的内容，决定你要采取的行动。
考虑因素包括：
1. 内容是否符合你的立场和情感倾向
2. 你的活跃度和交互偏好
3. 你的社会影响力
4. 你最近的交互历史

请输出你的决策。
"""
        return prompt

    async def decide_action(self, environment_content: Dict[str, Any]) -> Optional[UserAction]:
        """决定用户动作"""
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(environment_content)

            response = self.client.chat.completions.create(
                model=config.model.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.model.temperature,
                max_tokens=config.model.max_tokens
            )

            result = json.loads(response.choices[0].message.content)

            # 根据活跃度决定是否真的执行动作
            import random
            if random.random() > self.profile.activity_level:
                result["action"] = "ignore"

            if result["action"] == "ignore":
                return None

            # 创建动作对象
            action = UserAction(
                action_type=result["action"],
                target_id=environment_content.get("post_id", "unknown"),
                content=result.get("content", "")
            )

            # 记录到记忆中
            self.memory.add_interaction({
                "action": action.to_dict(),
                "environment": environment_content,
                "reasoning": result.get("reasoning", "")
            })

            return action

        except Exception as e:
            print(f"用户 {self.profile.user_id} 决策时出错: {e}")
            return None


class UserAgentManager:
    """用户智能体管理器"""

    def __init__(self):
        self.agents: Dict[str, UserAgent] = {}

    def add_agent(self, agent: UserAgent):
        """添加智能体"""
        self.agents[agent.profile.user_id] = agent

    def remove_agent(self, user_id: str):
        """移除智能体"""
        if user_id in self.agents:
            del self.agents[user_id]

    def get_agent(self, user_id: str) -> Optional[UserAgent]:
        """获取智能体"""
        return self.agents.get(user_id)

    async def simulate_batch_actions(self, user_ids: List[str], environment_content: Dict[str, Any]) -> List[UserAction]:
        """批量模拟用户行为 - 改进的并发实现"""
        max_concurrent = config.user_simulation.max_concurrent_users

        # 方法1: 使用asyncio.Semaphore控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def simulate_single_user(user_id: str) -> Optional[UserAction]:
            async with semaphore:
                agent = self.agents.get(user_id)
                if agent:
                    try:
                        return await agent.decide_action(environment_content)
                    except Exception as e:
                        print(f"用户 {user_id} 模拟失败: {e}")
                        return None
                return None

        # 创建任务并执行
        tasks = [simulate_single_user(user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤结果
        actions = []
        for i, result in enumerate(results):
            if isinstance(result, UserAction):
                actions.append(result)
            elif isinstance(result, Exception):
                print(f"用户 {user_ids[i]} 执行异常: {result}")

        return actions

    async def simulate_batch_actions_with_futures(self, user_ids: List[str], environment_content: Dict[str, Any]) -> List[UserAction]:
        """使用concurrent.futures的并发实现（适用于CPU密集型任务）"""
        max_workers = min(config.user_simulation.max_concurrent_users, len(user_ids))

        def sync_simulate_user(user_id: str) -> Optional[UserAction]:
            """同步版本的用户模拟（用于线程池）"""
            agent = self.agents.get(user_id)
            if not agent:
                return None

            try:
                # 创建新的OpenAI客户端（线程安全）
                client = OpenAI(
                    api_key=config.model.api_key,
                    base_url=config.model.base_url
                )

                # 构建提示词
                system_prompt = agent._build_system_prompt()
                user_prompt = agent._build_user_prompt(environment_content)

                # 同步API调用
                response = client.chat.completions.create(
                    model=config.model.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=config.model.temperature,
                    max_tokens=config.model.max_tokens
                )

                result = json.loads(response.choices[0].message.content)

                # 根据活跃度决定是否真的执行动作
                import random
                if random.random() > agent.profile.activity_level:
                    result["action"] = "ignore"

                if result["action"] == "ignore":
                    return None

                # 创建动作对象
                action = UserAction(
                    action_type=result["action"],
                    target_id=environment_content.get("post_id", "unknown"),
                    content=result.get("content", "")
                )

                # 记录到记忆中
                agent.memory.add_interaction({
                    "action": action.to_dict(),
                    "environment": environment_content,
                    "reasoning": result.get("reasoning", "")
                })

                return action

            except Exception as e:
                print(f"用户 {user_id} 模拟失败: {e}")
                return None

        # 使用线程池执行器
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [
                loop.run_in_executor(executor, sync_simulate_user, user_id)
                for user_id in user_ids
            ]

            # 等待所有任务完成
            results = await asyncio.gather(*futures, return_exceptions=True)

            actions = []
            for i, result in enumerate(results):
                if isinstance(result, UserAction):
                    actions.append(result)
                elif isinstance(result, Exception):
                    print(f"用户 {user_ids[i]} 执行异常: {result}")

        return actions

    async def simulate_batch_actions_chunked(self, user_ids: List[str], environment_content: Dict[str, Any]) -> List[UserAction]:
        """分块并发处理 - 适合大量用户的场景"""
        max_concurrent = config.user_simulation.max_concurrent_users
        chunk_size = max_concurrent

        all_actions = []

        # 将用户ID分块处理
        for i in range(0, len(user_ids), chunk_size):
            chunk = user_ids[i:i + chunk_size]
            print(f"处理用户块 {i//chunk_size + 1}/{(len(user_ids) + chunk_size - 1)//chunk_size}")

            # 处理当前块
            chunk_actions = await self.simulate_batch_actions(chunk, environment_content)
            all_actions.extend(chunk_actions)

            # 可选：块之间的延迟，避免API限流
            if i + chunk_size < len(user_ids):
                await asyncio.sleep(0.1)

        return all_actions

    def load_agents_from_config(self, config_path: str):
        """从配置文件加载智能体"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                users_data = json.load(f)

            for user_data in users_data:
                profile = UserProfile.from_dict(user_data)
                agent = UserAgent(profile)
                self.add_agent(agent)

        except Exception as e:
            print(f"加载用户配置时出错: {e}")

    def get_all_user_ids(self) -> List[str]:
        """获取所有用户ID"""
        return list(self.agents.keys())
