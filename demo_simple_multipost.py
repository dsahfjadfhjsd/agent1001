# -*- coding: utf-8 -*-
"""
简化的多帖子模拟演示
利用batch_id实现不同模拟批次的分离
"""

import asyncio
import os
import sys
import random
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(__file__))

# fmt: off
from UserAgent.user_profile_manager import UserProfileManager
from SimulateEnv import SimulationEngine, SimulationConfig
# fmt: on


class SimpleMultiPostDemo:
    """简化的多帖子模拟演示类"""

    def __init__(self, batch_id: str = None):
        """
        初始化多帖子演示

        Args:
            batch_id: 批次ID，用于区分不同的模拟
        """
        if batch_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"multipost_{timestamp}"

        self.batch_id = batch_id
        print(f"🚀 初始化多帖子模拟，批次ID: {self.batch_id}")

    def load_posts_from_csv(self, csv_path: str, num_posts: int = 5) -> List[Dict[str, Any]]:
        """
        从CSV文件加载帖子数据

        Args:
            csv_path: CSV文件路径
            num_posts: 要选择的帖子数量

        Returns:
            帖子列表
        """
        print(f"📂 从 {csv_path} 加载帖子...")

        # 读取CSV
        df = pd.read_csv(csv_path)
        print(f"   找到 {len(df)} 条帖子数据")

        # 随机选择指定数量的帖子
        if len(df) > num_posts:
            selected_df = df.sample(n=num_posts, random_state=42)
        else:
            selected_df = df

        # 转换为字典列表
        posts = []
        for _, row in selected_df.iterrows():
            post = {
                'content': row['content'],
                'platform': row.get('platform', 'unknown'),
                'title': row.get('title', ''),
                'original_likes': row.get('like_count', 0),
                'original_comments': row.get('comment_count', 0)
            }
            posts.append(post)

        print(f"✅ 选择了 {len(posts)} 个帖子进行模拟")
        return posts

    async def run_simulation(self,
                             posts: List[Dict[str, Any]],
                             config: SimulationConfig,
                             users_per_post: int = 10,
                             rounds_per_post: int = 2) -> Dict[str, Any]:
        """
        运行多帖子模拟

        Args:
            posts: 帖子列表
            config: 模拟配置
            users_per_post: 每个帖子的用户数
            rounds_per_post: 每个帖子的轮数

        Returns:
            模拟结果统计
        """
        print(f"\n🎯 开始多帖子模拟:")
        print(f"   帖子数量: {len(posts)}")
        print(f"   每帖用户数: {users_per_post}")
        print(f"   每帖轮数: {rounds_per_post}")
        print(f"   批次ID: {self.batch_id}")
        print(f"   行为概率: {config.action_probability}, 评论概率: {config.comment_probability}")
        print("-" * 50)

        # 创建引擎
        engine = SimulationEngine(config)

        # 生成用户
        user_manager = UserProfileManager()
        total_users = len(posts) * users_per_post
        print(f"👥 生成 {total_users} 个用户...")
        user_manager.generate_users(count=total_users, filename="demo_multi_post.csv")
        all_users = user_manager.get_all_users()

        # 模拟结果统计
        batch_results = {
            'batch_id': self.batch_id,
            'posts': [],
            'total_actions': 0,
            'total_users': total_users,
            'start_time': datetime.now().isoformat()
        }

        try:
            # 为每个帖子进行模拟
            for post_idx, post in enumerate(posts, 1):
                print(f"\n📝 模拟帖子 {post_idx}/{len(posts)}")
                print(f"   内容预览: {post['content'][:80]}...")

                # 为这个帖子分配用户
                start_idx = (post_idx - 1) * users_per_post
                end_idx = start_idx + users_per_post
                post_users = all_users[start_idx:end_idx]

                # 创建会话（使用batch_id）
                session_id = engine.create_session(
                    post_content=post['content'],
                    batch_id=self.batch_id
                )

                # 运行多轮模拟
                post_actions = []
                for round_num in range(1, rounds_per_post + 1):
                    print(f"   🔄 轮次 {round_num}/{rounds_per_post}")

                    round_actions = await engine.simulate_round_with_thinking(post_users)
                    if round_actions:
                        post_actions.extend(round_actions)

                        # 统计行为类型
                        behavior_stats = {}
                        for action in round_actions:
                            action_type = action.action_type.value
                            behavior_stats[action_type] = behavior_stats.get(action_type, 0) + 1

                        print(f"      - 生成 {len(round_actions)} 个行为: {behavior_stats}")
                    else:
                        print(f"      - 无行为生成")

                # 记录帖子结果
                post_result = {
                    'post_index': post_idx,
                    'session_id': session_id,
                    'content_preview': post['content'][:100],
                    'platform': post['platform'],
                    'users_count': len(post_users),
                    'rounds': rounds_per_post,
                    'actions_count': len(post_actions),
                    'action_types': {}
                }

                # 统计行为类型
                for action in post_actions:
                    action_type = action.action_type.value
                    post_result['action_types'][action_type] = post_result['action_types'].get(action_type, 0) + 1

                batch_results['posts'].append(post_result)
                batch_results['total_actions'] += len(post_actions)

                print(f"   ✅ 帖子 {post_idx} 完成: {len(post_actions)} 个行为")

        finally:
            await engine.close()

        # 完成统计
        batch_results['end_time'] = datetime.now().isoformat()
        batch_results['duration_minutes'] = (
            datetime.fromisoformat(batch_results['end_time']) -
            datetime.fromisoformat(batch_results['start_time'])
        ).total_seconds() / 60

        return batch_results

    def print_results(self, results: Dict[str, Any]):
        """打印模拟结果"""
        print(f"\n🎉 模拟完成！")
        print(f"   批次ID: {results['batch_id']}")
        print(f"   总用户数: {results['total_users']}")
        print(f"   总行为数: {results['total_actions']}")
        print(f"   耗时: {results['duration_minutes']:.1f} 分钟")

        print(f"\n📊 各帖子结果:")
        for post in results['posts']:
            print(f"   帖子 {post['post_index']}: {post['actions_count']} 个行为 - {post['action_types']}")

        print(f"\n📁 输出位置:")
        print(f"   数据文件: Output/exports/{results['batch_id']}/")
        print(f"   Prompt文件: Output/prompt_exports/{results['batch_id']}/")
        print(f"   用户记忆: UserAgent/user_memories/{results['batch_id']}/")


async def main():
    """主演示函数"""
    print("🧪 简化多帖子模拟演示")
    print("=" * 50)

    # 创建演示实例
    demo = SimpleMultiPostDemo()

    # 创建模拟配置
    config = SimulationConfig(
        max_concurrent_requests=3,
        action_probability=0.8,
        comment_probability=0.6,
        export_prompts=True,
        prompt_export_dir=f"Output/prompt_exports/{demo.batch_id}"
    )

    # 加载帖子数据
    posts = demo.load_posts_from_csv(
        csv_path="Data/integrated_data/XMSU7D_integrated_articles.csv",
        num_posts=3  # 演示用3个帖子
    )

    # 运行模拟
    results = await demo.run_simulation(
        posts=posts,
        config=config,      # 传入配置
        users_per_post=8,   # 每个帖子8个用户
        rounds_per_post=2   # 每个帖子2轮
    )

    # 显示结果
    demo.print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
