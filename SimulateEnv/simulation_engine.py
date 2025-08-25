# -*- coding: utf-8 -*-
"""
社交媒体交互模拟引擎

整合所有组件，提供完整的交互模拟功能
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# fmt:off
try:
    # 尝试相对导入
    from .interaction_core import InteractionEnvironment, UserAction
    from .data_storage import DataStorage
    from .user_behavior_simulator import UserBehaviorSimulator, SimulationConfig, UserThinkingResult
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from interaction_core import InteractionEnvironment, UserAction
    from data_storage import DataStorage
    from user_behavior_simulator import UserBehaviorSimulator, SimulationConfig, UserThinkingResult

# 导入用户记忆管理器
try:
    from UserAgent.user_memory_manager import UserMemoryManager, UserThinking
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from UserAgent.user_memory_manager import UserMemoryManager, UserThinking
# fmt:on


class SimulationEngine:
    """社交媒体交互模拟引擎"""

    def __init__(self, config: SimulationConfig = None, storage_dir: str = None):
        """
        初始化模拟引擎

        Args:
            config: 模拟配置
            storage_dir: 数据存储目录
        """
        self.config = config or SimulationConfig()
        self.storage = DataStorage(storage_dir)
        self.simulator = UserBehaviorSimulator(self.config)

        # 当前批次ID（用于分离不同模拟）
        self.current_batch_id: Optional[str] = None

        # 添加用户记忆管理器（初始化时不设置batch_id）
        memory_dir = os.path.join(storage_dir or 'data', 'user_memories') if storage_dir else None
        self.base_memory_dir = memory_dir
        self.memory_manager = None  # 将在create_session时初始化

        # 当前活跃会话
        self.current_session_id: Optional[str] = None
        self.current_environment: Optional[InteractionEnvironment] = None

    async def close(self):
        """关闭引擎和所有连接"""
        try:
            if self.simulator:
                await self.simulator.close()
        except:
            pass

    def create_session(self, post_content: str, session_id: str = None, batch_id: str = None) -> str:
        """
        创建新的模拟会话

        Args:
            post_content: 初始帖子内容
            session_id: 会话ID，如果不提供则自动生成
            batch_id: 批次ID，用于多批次模拟的分离

        Returns:
            会话ID
        """
        if session_id is None:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # session_id = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
            session_id = f"session_{uuid.uuid4().hex[:6]}"

        # 设置当前batch_id
        self.current_batch_id = batch_id

        # 初始化带batch_id的用户记忆管理器
        self.memory_manager = UserMemoryManager(self.base_memory_dir, batch_id)

        # 创建环境
        self.current_environment = InteractionEnvironment(post_content)
        self.current_session_id = session_id

        # 初始化保存时间戳
        self.storage._session_created_at = datetime.now().isoformat()

        # 保存初始状态
        self.storage.save_incremental_data(self.current_environment, session_id, batch_id)

        print(f"创建新会话: {session_id}")
        print(f"初始帖子: {post_content}")

        return session_id

    def load_session(self, session_id: str) -> bool:
        """
        加载已存在的会话

        Args:
            session_id: 会话ID

        Returns:
            是否加载成功
        """
        environment = self.storage.load_environment(session_id)
        if environment:
            self.current_environment = environment
            self.current_session_id = session_id
            print(f"成功加载会话: {session_id}")
            return True
        else:
            print(f"会话不存在: {session_id}")
            return False

    async def simulate_round_with_thinking(
        self,
        user_profiles: List[Dict[str, Any]],
        save_round_data: bool = True
    ) -> List[UserAction]:
        """
        模拟一轮用户交互（包含思考过程和认知变化）

        Args:
            user_profiles: 参与本轮的用户画像列表
            save_round_data: 是否保存本轮数据

        Returns:
            本轮生成的用户行为列表
        """
        if not self.current_environment or not self.current_session_id:
            raise ValueError("没有活跃会话，请先创建或加载会话")

        # 开始新一轮
        self.current_environment.start_new_round()
        round_number = self.current_environment.current_round

        print(f"\n=== 开始第 {round_number} 轮模拟（增强模式） ===")
        print(f"参与用户数: {len(user_profiles)}")
        print(f"📊 模拟进度：正在初始化用户记忆...")

        # 初始化用户记忆（如果还没有的话）
        print("📋 正在初始化用户记忆...")
        for user_profile in tqdm(user_profiles, desc="初始化记忆", leave=False, ncols=80):
            user_id = user_profile['user_id']
            if user_id not in self.memory_manager._memory_cache:
                self.memory_manager.initialize_user_memory(user_profile)

        # 获取当前环境状态
        env_state = self.current_environment.get_environment_state()

        # 为每个用户准备增强的画像（包含更新后的立场和情感值）
        print("🔄 正在准备用户增强画像...")
        enhanced_user_profiles = []
        for user_profile in tqdm(user_profiles, desc="准备画像", leave=False, ncols=80):
            current_profile = self.memory_manager.get_user_current_profile(user_profile['user_id'])
            if current_profile:
                enhanced_user_profiles.append(current_profile)
            else:
                enhanced_user_profiles.append(user_profile)

        # 并发模拟用户行为（使用增强的思考模式）
        start_time = datetime.now()
        thinking_results = await self._simulate_multiple_users_with_thinking(
            enhanced_user_profiles, env_state, round_number
        )
        simulation_time = (datetime.now() - start_time).total_seconds()

        print(f"模拟耗时: {simulation_time:.2f}秒")
        print(f"生成思考结果数: {len(thinking_results)}")

        # 处理思考结果
        print("📝 正在处理思考结果...")
        successful_actions = []
        for thinking_result in tqdm(thinking_results, desc="处理结果", leave=False, ncols=80):
            if thinking_result and thinking_result.action:
                # 将行为应用到环境
                if self.current_environment.add_action(thinking_result.action):
                    successful_actions.append(thinking_result.action)

                # 记录用户思考到记忆中
                user_thinking = UserThinking(
                    timestamp=datetime.now().isoformat(),
                    round_number=round_number,
                    content_seen=thinking_result.content_seen,
                    thinking_process=thinking_result.thinking_process,
                    stance_before=thinking_result.stance_before,
                    stance_after=thinking_result.stance_after,
                    sentiment_before=thinking_result.sentiment_before,
                    sentiment_after=thinking_result.sentiment_after,
                    action_taken=thinking_result.action.action_type.value if thinking_result.action else None,
                    action_content=thinking_result.action.content if thinking_result.action else None
                )

                self.memory_manager.add_user_thinking(thinking_result.user_id, user_thinking)

            elif thinking_result:
                # 即使没有行为，也要记录思考过程
                user_thinking = UserThinking(
                    timestamp=datetime.now().isoformat(),
                    round_number=round_number,
                    content_seen=thinking_result.content_seen,
                    thinking_process=thinking_result.thinking_process,
                    stance_before=thinking_result.stance_before,
                    stance_after=thinking_result.stance_after,
                    sentiment_before=thinking_result.sentiment_before,
                    sentiment_after=thinking_result.sentiment_after,
                    action_taken="no_action",
                    action_content=None
                )

                self.memory_manager.add_user_thinking(thinking_result.user_id, user_thinking)

        print(f"成功应用行为数: {len(successful_actions)}")

        # 增量保存数据（每轮结束后立即保存）
        if save_round_data:
            save_path = self.storage.save_incremental_data(self.current_environment, self.current_session_id, self.current_batch_id)
            print(f"数据已增量保存到: {save_path}")

        # 显示轮次结果（包含认知变化信息）
        self._print_enhanced_round_summary(round_number, successful_actions, thinking_results)

        return successful_actions

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """获取当前环境状态"""
        if self.current_environment:
            return self.current_environment.get_environment_state()
        return None

    def get_session_summary(self, session_id: str = None) -> Optional[Dict[str, Any]]:
        """获取会话摘要"""
        target_session = session_id or self.current_session_id
        if target_session:
            return self.storage.get_session_summary(target_session)
        return None

    def list_all_sessions(self) -> List[str]:
        """列出所有会话"""
        return self.storage.list_sessions()

    def export_session(self, session_id: str = None, export_dir: str = None) -> str:
        """导出会话数据"""
        target_session = session_id or self.current_session_id
        if not target_session:
            raise ValueError("没有指定会话ID")

        return self.storage.export_session_data(target_session, export_dir)

    def _print_round_summary(self, round_number: int, actions: List[UserAction]):
        """打印轮次摘要"""
        if not actions:
            print("本轮无用户行为")
            return

        # 统计行为类型
        action_stats = {}
        user_stats = {}

        for action in actions:
            action_type = action.action_type.value
            action_stats[action_type] = action_stats.get(action_type, 0) + 1
            user_stats[action.user_id] = user_stats.get(action.user_id, 0) + 1

        print(f"\n第 {round_number} 轮结果:")
        print(f"- 行为统计: {action_stats}")
        print(f"- 活跃用户: {len(user_stats)}人")

        # 显示部分具体行为
        print("- 部分行为详情:")
        for i, action in enumerate(actions[:5]):  # 只显示前5个
            content_info = f" (内容: {action.content[:20]}...)" if action.content else ""
            print(f"  {i+1}. {action.user_id}: {action.action_type.value}{content_info}")

        if len(actions) > 5:
            print(f"  ... 还有 {len(actions) - 5} 个行为")

    def _generate_final_summary(
        self,
        all_actions: List[UserAction],
        round_summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成最终摘要"""
        if not all_actions:
            return {
                'total_actions': 0,
                'unique_users': 0,
                'total_rounds': len(round_summaries),
                'action_types': {},
                'rounds': round_summaries
            }

        # 统计总体数据
        action_types = {}
        unique_users = set()

        for action in all_actions:
            action_types[action.action_type.value] = action_types.get(action.action_type.value, 0) + 1
            unique_users.add(action.user_id)

        return {
            'session_id': self.current_session_id,
            'total_actions': len(all_actions),
            'unique_users': len(unique_users),
            'total_rounds': len(round_summaries),
            'action_types': action_types,
            'rounds': round_summaries,
            'final_state': self.current_environment.get_environment_state() if self.current_environment else None
        }

    async def _simulate_multiple_users_with_thinking(
        self,
        user_profiles: List[Dict[str, Any]],
        environment_state: Dict[str, Any],
        round_number: int
    ) -> List[UserThinkingResult]:
        """
        并发模拟多个用户的思考过程

        Args:
            user_profiles: 用户画像列表
            environment_state: 当前环境状态
            round_number: 当前轮次

        Returns:
            用户思考结果列表
        """
        tasks = []
        for user_profile in user_profiles:
            # 获取用户的历史记忆
            user_memory = self.memory_manager.get_user_recent_interactions(
                user_profile['user_id'], max_count=3
            )

            # 转换记忆格式为字典
            memory_dicts = []
            for memory in user_memory:
                memory_dict = {
                    'round_number': memory.round_number,
                    'thinking_process': memory.thinking_process,
                    'stance_before': memory.stance_before,
                    'stance_after': memory.stance_after,
                    'sentiment_before': memory.sentiment_before,
                    'sentiment_after': memory.sentiment_after,
                    'action_taken': memory.action_taken,
                    'action_content': memory.action_content
                }
                memory_dicts.append(memory_dict)

            task = self.simulator.simulate_user_behavior_with_thinking(
                user_profile, environment_state, round_number, memory_dicts
            )
            tasks.append(task)

        # 使用tqdm显示进度条
        print(f"🧠 正在模拟 {len(user_profiles)} 个用户的思考过程...")

        # 包装任务以处理异常
        async def safe_task(task, pbar):
            try:
                result = await task
                pbar.update(1)
                return result
            except Exception as e:
                pbar.update(1)
                return e

        # 创建进度条
        with tqdm(total=len(tasks), desc="用户思考模拟", unit="用户", ncols=80) as pbar:
            safe_tasks = [safe_task(task, pbar) for task in tasks]
            results = await asyncio.gather(*safe_tasks)

        # 过滤出成功的思考结果
        thinking_results = []
        successful_count = 0
        error_count = 0

        for result in results:
            if isinstance(result, UserThinkingResult):
                thinking_results.append(result)
                successful_count += 1
            elif isinstance(result, Exception):
                error_count += 1
                print(f"⚠️  用户思考模拟异常: {result}")
            else:
                # None result (用户未采取行动)
                pass

        print(f"✅ 完成思考模拟 - 成功: {successful_count}, 错误: {error_count}, 无思考: {len(user_profiles) - successful_count - error_count}")
        return thinking_results

    def _print_enhanced_round_summary(
        self,
        round_number: int,
        actions: List[UserAction],
        thinking_results: List[UserThinkingResult]
    ):
        """
        打印增强的轮次结果摘要

        Args:
            round_number: 轮次号
            actions: 成功的行为列表
            thinking_results: 思考结果列表
        """
        if not actions:
            print("本轮无用户行为")
            return

        # 统计行为类型
        action_types = {}
        for action in actions:
            action_type = action.action_type.value
            action_types[action_type] = action_types.get(action_type, 0) + 1

        # 统计认知变化
        stance_changes = []
        sentiment_changes = []
        for thinking_result in thinking_results:
            if thinking_result:
                stance_change = thinking_result.stance_after - thinking_result.stance_before
                sentiment_change = thinking_result.sentiment_after - thinking_result.sentiment_before
                if abs(stance_change) > 0.01:  # 只记录有意义的变化
                    stance_changes.append(stance_change)
                if abs(sentiment_change) > 0.01:
                    sentiment_changes.append(sentiment_change)

        print(f"\n第 {round_number} 轮结果:")
        print(f"- 行为统计: {action_types}")
        print(f"- 活跃用户: {len(set(action.user_id for action in actions))}人")

        if stance_changes:
            avg_stance_change = sum(stance_changes) / len(stance_changes)
            print(f"- 平均立场变化: {avg_stance_change:.3f} ({len(stance_changes)}人有变化)")

        if sentiment_changes:
            avg_sentiment_change = sum(sentiment_changes) / len(sentiment_changes)
            print(f"- 平均情感变化: {avg_sentiment_change:.3f} ({len(sentiment_changes)}人有变化)")

        # 显示部分行为详情
        print("- 部分行为详情:")
        for i, action in enumerate(actions[:3], 1):
            action_desc = action.action_type.value
            if action.content:
                content_preview = action.content[:30] + "..." if len(action.content) > 30 else action.content
                action_desc += f" (内容: {content_preview})"
            print(f"  {i}. {action.user_id[:20]}...: {action_desc}")

    def get_user_cognition_changes(self) -> Dict[str, Any]:
        """
        获取所有用户的认知变化统计

        Returns:
            认知变化统计信息
        """
        return self.memory_manager.get_statistics()
