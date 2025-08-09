"""
系统集成模块 - 整合用户模拟、环境管理和分发评估
"""
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from UserAgent.user_agent import UserAgentManager
from UserAgent.user_profile import UserProfile
from SimulateEnv.environment_manager import EnvironmentManager
from SimulateEnv.environment_models import Post
from DisAgent.distribution_agent import DistributionManager
from Config.settings import config


class CognitiveGuidanceSystem:
    """认知引导系统 - 主系统类"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())

        # 初始化三个核心模块
        self.user_manager = UserAgentManager()
        self.environment = EnvironmentManager(session_id=self.session_id)
        self.distribution_manager = DistributionManager()

        # 系统状态
        self.is_initialized = False
        self.current_post: Optional[Post] = None

    def initialize_system(self, user_config_path: str = None):
        """初始化系统"""
        if user_config_path is None:
            user_config_path = "Config/sample_users.json"

        # 加载用户配置
        self.user_manager.load_agents_from_config(user_config_path)
        self.environment.user_manager = self.user_manager

        # 创建默认分发策略
        default_strategy = self.distribution_manager.create_default_strategy()
        self.distribution_manager.add_strategy(default_strategy)

        self.is_initialized = True
        print(f"系统已初始化，会话ID: {self.session_id}")
        print(f"已加载 {len(self.user_manager.agents)} 个用户智能体")

    async def start_simulation(self, post_data: Dict[str, Any], max_rounds: int = 3,
                               concurrency_method: str = None) -> Dict[str, Any]:
        """开始模拟过程

        Args:
            post_data: 帖子数据
            max_rounds: 最大轮次
            concurrency_method: 并发方法 ("asyncio", "futures", "chunked")
        """
        if not self.is_initialized:
            raise ValueError("系统未初始化，请先调用 initialize_system()")

        if concurrency_method is None:
            concurrency_method = config.user_simulation.concurrency_method

        print(f"开始模拟，最大轮次: {max_rounds}，并发方法: {concurrency_method}")

        # 1. 创建帖子
        post = Post(
            post_id=post_data.get("post_id", str(uuid.uuid4())),
            title=post_data.get("title", ""),
            content=post_data.get("content", ""),
            author=post_data.get("author", "system"),
            timestamp=datetime.now(),
            tags=post_data.get("tags", []),
            media_urls=post_data.get("media_urls", [])
        )
        self.current_post = post

        # 2. 初始化环境
        self.environment.initialize_environment(post)

        # 3. 执行分发决策
        available_users = [agent.profile for agent in self.user_manager.agents.values()]
        distribution_result = await self.distribution_manager.execute_distribution(
            content=post.to_dict(),
            available_users=available_users
        )

        print(f"分发决策完成，目标用户: {len(distribution_result.target_users)} 人")

        # 4. 执行多轮模拟
        simulation_results = []
        current_users = distribution_result.target_users.copy()

        for round_num in range(1, max_rounds + 1):
            print(f"执行第 {round_num} 轮模拟...")

            # 模拟用户交互
            round_result = await self.environment.simulate_round(
                participating_users=current_users,
                concurrent=True,
                concurrency_method=concurrency_method
            )
            simulation_results.append(round_result)

            # 动态调整参与用户（可以添加新用户或移除不活跃用户）
            if round_num < max_rounds:
                current_users = self._update_participating_users(current_users, round_result)

        # 5. 评估分发效果
        environment_data = self.environment.get_environment_for_display()
        final_metrics = await self.distribution_manager.evaluate_distribution_effectiveness(
            distribution_result.distribution_id,
            environment_data
        )

        # 6. 生成综合报告
        final_report = {
            "session_id": self.session_id,
            "post": post.to_dict(),
            "distribution_result": distribution_result.to_dict(),
            "simulation_rounds": simulation_results,
            "final_metrics": final_metrics.to_dict(),
            "environment_data": environment_data,
            "timestamp": datetime.now().isoformat()
        }

        print("模拟完成，正在生成报告...")

        # 7. 保存结果
        report_path = self._save_simulation_report(final_report)
        html_path = self.environment.export_html_report()

        print(f"报告已保存: {report_path}")
        print(f"HTML报告: {html_path}")

        return final_report

    def _update_participating_users(self, current_users: List[str], round_result: Dict[str, Any]) -> List[str]:
        """动态更新参与用户列表"""
        # 简单策略：保持活跃用户，可能添加新用户
        updated_users = current_users.copy()

        # 可以根据round_result中的参与情况调整用户列表
        # 这里使用简单的策略：如果参与度低，添加更多用户
        engagement = round_result.get("actions_taken", 0)
        if engagement < len(current_users) * 0.3:  # 如果参与度低于30%
            # 随机添加一些新用户
            all_users = self.user_manager.get_all_user_ids()
            for user_id in all_users:
                if user_id not in updated_users and len(updated_users) < 10:
                    updated_users.append(user_id)
                    break

        return updated_users

    def _save_simulation_report(self, report: Dict[str, Any]) -> str:
        """保存模拟报告"""
        filename = f"simulation_report_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"{config.environment.save_path}/{filename}"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return filepath

    async def evaluate_strategy_and_optimize(self) -> Dict[str, Any]:
        """评估策略并提供优化建议"""
        if not self.distribution_manager.results_history:
            return {"message": "没有足够的历史数据进行评估"}

        # 获取所有策略的评估结果
        strategy_evaluations = {}

        for strategy_id, strategy in self.distribution_manager.strategies.items():
            # 获取该策略的结果
            strategy_results = [
                result for result in self.distribution_manager.results_history
                if result.strategy_id == strategy_id
            ]

            if strategy_results:
                evaluation = await self.distribution_manager.evaluator.evaluate_strategy_performance(
                    strategy, strategy_results
                )
                strategy_evaluations[strategy_id] = evaluation

        return {
            "session_id": self.session_id,
            "strategy_evaluations": strategy_evaluations,
            "timestamp": datetime.now().isoformat()
        }

    async def run_batch_experiments(self, posts_data: List[Dict[str, Any]],
                                    strategies: List[str] = None) -> Dict[str, Any]:
        """批量实验不同内容和策略的组合"""
        experiments = []

        for i, post_data in enumerate(posts_data):
            print(f"执行实验 {i+1}/{len(posts_data)}")

            if strategies:
                for strategy_id in strategies:
                    if strategy_id in self.distribution_manager.strategies:
                        # 重置环境
                        self.environment = EnvironmentManager(session_id=f"{self.session_id}_exp_{i}_{strategy_id}")
                        self.environment.user_manager = self.user_manager

                        # 执行模拟
                        result = await self.start_simulation(post_data, max_rounds=2)
                        result["experiment_id"] = f"exp_{i}_{strategy_id}"
                        experiments.append(result)
            else:
                # 使用默认策略
                result = await self.start_simulation(post_data, max_rounds=2)
                result["experiment_id"] = f"exp_{i}_default"
                experiments.append(result)

        # 分析实验结果
        analysis = self._analyze_batch_results(experiments)

        return {
            "session_id": self.session_id,
            "experiments": experiments,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_batch_results(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析批量实验结果"""
        if not experiments:
            return {}

        # 计算平均指标
        total_metrics = {}
        for exp in experiments:
            metrics = exp.get("final_metrics", {})
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)

        avg_metrics = {}
        for key, values in total_metrics.items():
            avg_metrics[key] = sum(values) / len(values) if values else 0

        # 找出最佳实验
        best_experiment = max(experiments,
                              key=lambda x: x.get("final_metrics", {}).get("engagement_rate", 0))

        return {
            "total_experiments": len(experiments),
            "average_metrics": avg_metrics,
            "best_experiment_id": best_experiment.get("experiment_id"),
            "best_engagement_rate": best_experiment.get("final_metrics", {}).get("engagement_rate", 0)
        }

# 便捷函数


async def quick_simulation(post_content: str, post_title: str = "", max_rounds: int = 3) -> Dict[str, Any]:
    """快速开始一个模拟"""
    system = CognitiveGuidanceSystem()
    system.initialize_system()

    post_data = {
        "title": post_title or "模拟帖子",
        "content": post_content,
        "author": "system"
    }

    return await system.start_simulation(post_data, max_rounds)

# 示例使用
if __name__ == "__main__":
    # 示例：运行一个简单的模拟
    async def main():
        result = await quick_simulation(
            post_content="人工智能技术的发展将如何影响我们的未来工作？大家怎么看？",
            post_title="AI与未来工作讨论",
            max_rounds=3
        )
        print("模拟完成！")
        print(f"最终参与率: {result['final_metrics']['engagement_rate']:.2f}")

    asyncio.run(main())
