# -*- coding: utf-8 -*-
"""
智能多轮社交媒体模拟演示

基于内容分发算法，实现更真实的多轮交互模拟：
1. 初始化所有帖子的post_id
2. 使用分发算法智能选择每轮的帖子和用户
3. 基于热门度和随机性进行内容分发
4. 支持多轮模拟，每轮结果影响下一轮的分发策略
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(__file__))

# fmt: off
from UserAgent.user_profile_manager import UserProfileManager
from SimulateEnv import SimulationEngine, SimulationConfig
from DisAgent.content_distributor import ContentDistributor
# fmt: on


class IntelligentMultiRoundSimulation:
    """智能多轮社交媒体模拟类"""

    def __init__(self, batch_id: str = None):
        """
        初始化多轮模拟

        Args:
            batch_id: 批次ID，用于区分不同的模拟实验
        """
        if batch_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"intelligent_sim_{timestamp}"

        self.batch_id = batch_id
        self.distributor = ContentDistributor(batch_id)
        self.simulation_results = {
            'batch_id': batch_id,
            'rounds': [],
            'total_stats': {
                'total_posts_used': 0,
                'total_users_involved': 0,
                'total_interactions': 0,
                'simulation_duration': 0
            }
        }

        print(f"🚀 智能多轮模拟初始化完成 - 批次: {batch_id}")

    def load_posts_from_csv(self, csv_path: str, max_posts: int = 20) -> List[Dict[str, Any]]:
        """
        从CSV文件加载帖子数据

        Args:
            csv_path: CSV文件路径
            max_posts: 最大帖子数量

        Returns:
            帖子列表
        """
        print(f"📂 从 {csv_path} 加载帖子...")

        # 读取CSV
        df = pd.read_csv(csv_path)
        print(f"   找到 {len(df)} 条帖子数据")

        # 选择帖子
        if len(df) > max_posts:
            selected_df = df.sample(n=max_posts, random_state=42)
        else:
            selected_df = df

        # 转换为字典列表
        posts = []
        for _, row in selected_df.iterrows():
            post = {
                'content': row['content'],
                'platform': row.get('platform', 'unknown'),
                'title': row.get('title', ''),
                # 'original_likes': row.get('like_count', 0),
                # 'original_comments': row.get('comment_count', 0)
                'original_likes': 0,
                'original_comments': 0
            }
            posts.append(post)

        print(f"✅ 加载了 {len(posts)} 个帖子用于模拟")
        return posts

    def initialize_simulation(self, posts: List[Dict[str, Any]], total_users: int = 100, user_manager: UserProfileManager = None):
        """
        初始化模拟环境

        Args:
            posts: 帖子列表
            total_users: 总用户数量
            user_manager: 用户管理器
        """
        print(f"\n🎯 初始化模拟环境...")

        # 生成用户
        if user_manager is None:
            user_manager = UserProfileManager()
            user_manager.generate_users(count=total_users, filename=f"intelligent_sim_{self.batch_id}.csv")
            all_users = user_manager.get_all_users()
        else:
            all_users = user_manager.get_all_users()
            if len(all_users) > total_users:
                all_users = all_users[:total_users]

        # 使用分发器初始化批次
        self.distributor.initialize_batch(posts, all_users)

        print(f"✅ 模拟环境初始化完成")
        print(f"   总帖子数: {len(posts)}")
        print(f"   总用户数: {len(all_users)}")

    async def run_round(self, round_number: int,
                        posts_per_round: int = 5,
                        users_per_post: int = 10,
                        rounds_per_post: int = 2,
                        config: SimulationConfig = None) -> Dict[str, Any]:
        """
        运行单轮模拟

        Args:
            round_number: 轮次号
            posts_per_round: 每轮参与的帖子数量
            users_per_post: 每个帖子分配的用户数量
            rounds_per_post: 每个帖子内的交互轮数
            config: 模拟配置

        Returns:
            轮次结果
        """
        print(f"\n🎮 === 开始第 {round_number} 轮模拟 ===")

        # 生成分发计划
        distribution = self.distributor.generate_round_distribution(
            round_number=round_number,
            posts_per_round=posts_per_round,
            users_per_post=users_per_post,
            hot_post_ratio=0.4 if round_number > 1 else 0.0
        )

        if config is None:
            config = SimulationConfig(
                max_concurrent_requests=3,
                action_probability=0.8,
                comment_probability=0.6,
                export_prompts=True,
                prompt_export_dir=f"Output/prompt_exports/{self.batch_id}/round_{round_number}"
            )

        if config.prompt_export_dir is None and config.export_prompts:
            config.prompt_export_dir = f"Output/prompt_exports/intel_multi_{self.batch_id}/round_{round_number}"

        # 创建引擎
        engine = SimulationEngine(config)

        # 轮次结果统计（精简版）
        round_results = {
            'round_number': round_number,
            'posts': [],
            'total_actions': 0,
            'start_time': datetime.now().isoformat()
        }

        try:
            # 为每个分发的帖子进行模拟
            for post_idx, (post_id, post_dist) in enumerate(distribution['posts'].items(), 1):
                print(f"\n📝 模拟帖子 {post_idx}/{len(distribution['posts'])}: {post_id}")

                # 获取帖子内容
                post_data = self.distributor.distribution_plan['posts'][post_id]
                post_content = post_data['content']
                print(f"   内容预览: {post_content[:80]}...")

                # 获取分配的用户
                assigned_user_ids = post_dist['assigned_users']
                user_profiles = []
                for user_id in assigned_user_ids:
                    user_data = self.distributor.distribution_plan['users'][user_id]
                    user_profiles.append(user_data['profile'])

                print(f"   分配用户: {len(user_profiles)} 个")
                print(f"   选择原因: {post_dist['selection_reason']}")

                # 创建会话
                session_post_id = engine.create_session(
                    post_content=post_content,
                    post_id=post_id,
                    batch_id=self.batch_id
                )

                # 运行帖子内的多轮交互
                post_actions = []
                for inner_round in range(1, rounds_per_post + 1):
                    print(f"   🔄 内部轮次 {inner_round}/{rounds_per_post}")

                    inner_actions = await engine.simulate_round_with_thinking(user_profiles)
                    if inner_actions:
                        post_actions.extend(inner_actions)

                        # 统计行为类型
                        behavior_stats = {}
                        for action in inner_actions:
                            action_type = action.action_type.value
                            behavior_stats[action_type] = behavior_stats.get(action_type, 0) + 1

                        print(f"      - 生成 {len(inner_actions)} 个行为: {behavior_stats}")
                    else:
                        print(f"      - 无行为生成")

                # 记录帖子结果（精简版）
                post_result = {
                    'post_id': post_id,
                    'users_count': len(user_profiles),
                    'actions_count': len(post_actions),
                    'action_types': {}
                }

                # 统计行为类型
                for action in post_actions:
                    action_type = action.action_type.value
                    post_result['action_types'][action_type] = post_result['action_types'].get(action_type, 0) + 1

                round_results['posts'].append(post_result)
                round_results['total_actions'] += len(post_actions)

                print(f"   ✅ 帖子 {post_id} 完成: {len(post_actions)} 个行为")

        finally:
            await engine.close()

        # 完成轮次统计
        round_results['end_time'] = datetime.now().isoformat()
        round_results['duration_minutes'] = (
            datetime.fromisoformat(round_results['end_time']) -
            datetime.fromisoformat(round_results['start_time'])
        ).total_seconds() / 60

        # 更新分发器的轮次结果（用于下一轮的热门度计算）
        self.distributor.update_round_results(round_number, round_results)

        # 保存轮次结果
        self.simulation_results['rounds'].append(round_results)
        self._save_simulation_results()

        print(f"✅ 第 {round_number} 轮完成")
        print(f"   参与帖子: {len(round_results['posts'])} 个")
        print(f"   总行为数: {round_results['total_actions']} 个")
        print(f"   耗时: {round_results['duration_minutes']:.1f} 分钟")

        return round_results

    async def run_multiple_rounds(self, num_rounds: int = 3,
                                  posts_per_round: int = 5,
                                  users_per_post: int = 10,
                                  rounds_per_post: int = 2,
                                  config: SimulationConfig = None) -> Dict[str, Any]:
        """
        运行多轮模拟

        Args:
            num_rounds: 总轮数
            posts_per_round: 每轮参与的帖子数量
            users_per_post: 每个帖子分配的用户数量
            rounds_per_post: 每个帖子内的交互轮数
            config: 模拟配置

        Returns:
            完整的模拟结果
        """
        print(f"\n🎯 开始 {num_rounds} 轮智能模拟")
        print("=" * 60)

        start_time = datetime.now()

        for round_num in range(1, num_rounds + 1):
            await self.run_round(
                round_number=round_num,
                posts_per_round=posts_per_round,
                users_per_post=users_per_post,
                rounds_per_post=rounds_per_post,
                config=config
            )

            # 显示轮间摘要
            if round_num < num_rounds:
                self._print_inter_round_summary(round_num)

        # 完成总体统计
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 60

        # 计算参与的总用户数
        total_users_set = set()
        for round_data in self.distributor.distribution_plan['rounds']:
            for post_id, post_data in round_data['posts'].items():
                total_users_set.update(post_data['assigned_users'])

        self.simulation_results['total_stats'].update({
            'simulation_duration': total_duration,
            'total_posts_used': len(set(p['post_id'] for r in self.simulation_results['rounds'] for p in r['posts'])),
            'total_users_involved': len(total_users_set),
            'total_interactions': sum(r['total_actions'] for r in self.simulation_results['rounds'])
        })

        self._save_simulation_results()
        return self.simulation_results

    def print_final_summary(self):
        """打印最终模拟摘要"""
        print(f"\n🎉 智能多轮模拟完成！")
        print("=" * 60)

        results = self.simulation_results
        stats = results['total_stats']

        print(f"📋 总体统计:")
        print(f"   批次ID: {results['batch_id']}")
        print(f"   总轮数: {len(results['rounds'])}")
        print(f"   使用帖子: {stats['total_posts_used']} 个")
        print(f"   参与用户: {stats['total_users_involved']} 个")
        print(f"   总交互数: {stats['total_interactions']} 次")
        print(f"   总耗时: {stats['simulation_duration']:.1f} 分钟")

        print(f"\n📊 各轮详情:")
        for round_data in results['rounds']:
            round_num = round_data['round_number']
            posts_count = len(round_data['posts'])
            actions_count = round_data['total_actions']
            duration = round_data['duration_minutes']

            print(f"   第 {round_num} 轮: {posts_count} 帖子, {actions_count} 行为, {duration:.1f}分钟")

        # 显示热门帖子
        hot_posts = self.distributor.get_hot_posts(5)
        if hot_posts:
            print(f"\n🔥 热门帖子 Top 5:")
            for i, post in enumerate(hot_posts, 1):
                print(f"   {i}. {post['post_id']}: 热度 {post['heat_score']:.1f}, "
                      f"{post['total_interactions']} 交互, {post['unique_users']} 用户")
                print(f"      内容: {post['content_preview']}")

        # 显示分发摘要
        dist_summary = self.distributor.get_distribution_summary()
        unused_count = dist_summary['unused_posts']
        if unused_count > 0:
            print(f"\n📦 未使用帖子: {unused_count} 个 (可用于后续轮次)")

        print(f"\n📁 输出位置:")
        print(f"   模拟数据: Output/exports/{results['batch_id']}/")
        print(f"   分发计划: Output/exports/{results['batch_id']}/distribution_plan.json")
        print(f"   模拟结果: Output/exports/{results['batch_id']}/simulation_results.json")
        print(f"   用户记忆: UserAgent/user_memories/{results['batch_id']}/")

    def _print_inter_round_summary(self, completed_round: int):
        """打印轮间摘要"""
        print(f"\n📈 第 {completed_round} 轮结束后状态:")

        # 显示当前热门帖子
        hot_posts = self.distributor.get_hot_posts(3)
        if hot_posts:
            print(f"   🔥 当前热门帖子:")
            for post in hot_posts:
                print(f"      - {post['post_id']}: 热度 {post['heat_score']:.1f}")

        # 显示剩余帖子
        unused_count = len(self.distributor.get_unused_posts())
        print(f"   📦 剩余未使用帖子: {unused_count} 个")

        print(f"   ⏳ 准备下一轮...")

    def _save_simulation_results(self):
        """保存模拟结果"""
        results_file = self.distributor.batch_dir / "simulation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.simulation_results, f, indent=2, ensure_ascii=False)


async def main():
    """主演示函数"""
    print("🧠 智能多轮社交媒体模拟演示")
    print("=" * 60)

    # 创建模拟实例
    simulation = IntelligentMultiRoundSimulation()

    # 加载帖子数据
    posts = simulation.load_posts_from_csv(
        csv_path="Data/integrated_data/XMSU7D_integrated_articles.csv",
        max_posts=3  # 加载3个帖子，支持多轮选择
    )

    # 初始化模拟环境
    user_manager = UserProfileManager()
    users_count = user_manager.load_users_from_file("demo_users_enhanced.csv")
    simulation.initialize_simulation(
        posts=posts,
        total_users=20,  # 20个用户
        user_manager=user_manager
    )

    # 创建模拟配置
    config = SimulationConfig(
        max_concurrent_requests=4,
        action_probability=0.7,
        comment_probability=0.5,
        export_prompts=True,
        prompt_export_dir=f"Output/prompt_exports/intel_multi_{simulation.batch_id}"
    )

    # 运行3轮模拟
    results = await simulation.run_multiple_rounds(
        num_rounds=2,         # 总共运行次数
        posts_per_round=2,    # 每轮帖子数
        users_per_post=4,     # 每个帖子每轮曝光用户数
        rounds_per_post=1,    # 每个帖子每轮和曝光用户交互次数
        config=config         # 模拟配置
    )

    # 显示最终摘要
    simulation.print_final_summary()


if __name__ == "__main__":
    asyncio.run(main())
