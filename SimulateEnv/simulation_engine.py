"""
社交媒体交互模拟引擎

整合所有组件，提供完整的交互模拟功能
"""

from .user_behavior_simulator import UserBehaviorSimulator, SimulationConfig
from .data_storage import DataStorage
from .interaction_core import InteractionEnvironment, UserAction
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


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

        # 当前活跃会话
        self.current_session_id: Optional[str] = None
        self.current_environment: Optional[InteractionEnvironment] = None

    def create_session(self, post_content: str, session_id: str = None) -> str:
        """
        创建新的模拟会话

        Args:
            post_content: 初始帖子内容
            session_id: 会话ID，如果不提供则自动生成

        Returns:
            会话ID
        """
        if session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"

        # 创建环境
        self.current_environment = InteractionEnvironment(post_content)
        self.current_session_id = session_id

        # 保存初始状态
        self.storage.save_environment(self.current_environment, session_id)

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

    async def simulate_round(
        self,
        user_profiles: List[Dict[str, Any]],
        save_round_data: bool = True
    ) -> List[UserAction]:
        """
        模拟一轮用户交互

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

        print(f"\n=== 开始第 {round_number} 轮模拟 ===")
        print(f"参与用户数: {len(user_profiles)}")

        # 获取当前环境状态
        env_state = self.current_environment.get_environment_state()

        # 并发模拟用户行为
        start_time = datetime.now()
        actions = await self.simulator.simulate_multiple_users(
            user_profiles, env_state, round_number
        )
        simulation_time = (datetime.now() - start_time).total_seconds()

        print(f"模拟耗时: {simulation_time:.2f}秒")
        print(f"生成行为数: {len(actions)}")

        # 将行为应用到环境
        successful_actions = []
        for action in actions:
            if self.current_environment.add_action(action):
                successful_actions.append(action)
            else:
                print(f"行为应用失败: {action.user_id} - {action.action_type.value}")

        print(f"成功应用行为数: {len(successful_actions)}")

        # 保存数据
        if save_round_data:
            self.storage.save_environment(self.current_environment, self.current_session_id)
            if successful_actions:
                self.storage.save_round_actions(
                    successful_actions, self.current_session_id, round_number
                )

        # 显示轮次结果
        self._print_round_summary(round_number, successful_actions)

        return successful_actions

    async def run_simulation(
        self,
        user_profiles: List[Dict[str, Any]],
        num_rounds: int = 3,
        users_per_round: int = None,
        randomize_users: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整的多轮模拟

        Args:
            user_profiles: 用户画像列表
            num_rounds: 模拟轮数
            users_per_round: 每轮参与的用户数，如果不指定则使用所有用户
            randomize_users: 是否随机选择参与用户

        Returns:
            模拟结果摘要
        """
        if not self.current_environment or not self.current_session_id:
            raise ValueError("没有活跃会话，请先创建或加载会话")

        print(f"\n{'='*50}")
        print(f"开始多轮模拟 - 会话ID: {self.current_session_id}")
        print(f"总轮数: {num_rounds}")
        print(f"用户池大小: {len(user_profiles)}")
        print(f"{'='*50}")

        all_actions = []
        round_summaries = []

        for round_num in range(num_rounds):
            # 选择参与本轮的用户
            if users_per_round and users_per_round < len(user_profiles):
                if randomize_users:
                    import random
                    round_users = random.sample(user_profiles, users_per_round)
                else:
                    round_users = user_profiles[:users_per_round]
            else:
                round_users = user_profiles

            # 模拟本轮
            round_actions = await self.simulate_round(round_users)
            all_actions.extend(round_actions)

            # 记录本轮摘要
            round_summary = {
                'round_number': self.current_environment.current_round,
                'participants': len(round_users),
                'actions_generated': len(round_actions),
                'action_types': {},
                'active_users': len(set(action.user_id for action in round_actions))
            }

            # 统计行为类型
            for action in round_actions:
                action_type = action.action_type.value
                round_summary['action_types'][action_type] = round_summary['action_types'].get(action_type, 0) + 1

            round_summaries.append(round_summary)

            # 轮间休息（可以添加延迟模拟真实时间）
            if round_num < num_rounds - 1:
                await asyncio.sleep(0.1)  # 短暂延迟

        # 生成最终摘要
        final_summary = self._generate_final_summary(all_actions, round_summaries)

        # 保存最终状态
        self.storage.save_environment(self.current_environment, self.current_session_id)

        print(f"\n{'='*50}")
        print("模拟完成！")
        print(f"总行为数: {len(all_actions)}")
        print(f"参与用户数: {final_summary['unique_users']}")
        print(f"{'='*50}")

        return final_summary

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


if __name__ == "__main__":
    # 测试代码
    async def test_simulation():
        # 创建模拟引擎
        engine = SimulationEngine()

        # 创建会话
        session_id = engine.create_session("人工智能技术的发展对就业市场会产生什么影响？")

        # 模拟用户画像（实际使用时应该从UserAgent模块加载）
        test_users = [
            {
                'user_id': 'user_001',
                'age_group': '25-35',
                'gender': '男',
                'occupation': '软件工程师',
                'activity_level': '高',
                'stance': '支持',
                'sentiment': '积极'
            },
            {
                'user_id': 'user_002',
                'age_group': '35-45',
                'gender': '女',
                'occupation': '教师',
                'activity_level': '中等',
                'stance': '中立',
                'sentiment': '中立'
            },
            {
                'user_id': 'user_003',
                'age_group': '18-25',
                'gender': '男',
                'occupation': '学生',
                'activity_level': '中等',
                'stance': '反对',
                'sentiment': '消极'
            }
        ]

        # 运行模拟
        try:
            summary = await engine.run_simulation(
                user_profiles=test_users,
                num_rounds=2,
                users_per_round=2,
                randomize_users=True
            )

            print("\n最终摘要:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"模拟过程出错: {e}")
            # 使用备用方案
            print("使用备用模拟方案...")

    # 运行测试
    # asyncio.run(test_simulation())
