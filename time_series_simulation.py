# -*- coding: utf-8 -*-
"""
时间序列社交媒体模拟测试

按真实时间顺序模拟社交媒体发展，与真实情况对比分析
"""

import asyncio
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import uuid

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(__file__))

# fmt: off
from UserAgent.user_profile_manager import UserProfileManager
from SimulateEnv import SimulationEngine, SimulationConfig
from DisAgent.content_distributor import ContentDistributor
# fmt: on

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TimeSeriesSimulation:
    """时间序列社交媒体模拟类"""

    def __init__(self, batch_id: str = None):
        """
        初始化时间序列模拟

        Args:
            batch_id: 批次ID
        """
        if batch_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"timeseries_sim_{timestamp}"

        self.batch_id = batch_id
        self.distributor = ContentDistributor(batch_id)
        self.output_dir = "Output"  # 设置输出目录

        # 时间序列相关参数
        self.start_time = None
        self.time_step_hours = 6  # 每6小时为一个时间步
        self.current_time = None

        # 数据存储
        self.all_posts = []
        self.time_points = []
        self.stance_history = []
        self.sentiment_history = []
        self.interaction_counts = []
        self.round_active_users = {}  # 记录每轮参与的用户ID
        self.round_results = {}  # 记录每轮的模拟结果

    def load_and_prepare_posts(self, csv_path: str, start_date: str = "2025-03-31 24:00",
                               sample_ratio: float = 0.3) -> List[Dict[str, Any]]:
        """
        加载并准备按时间排序的帖子数据

        Args:
            csv_path: CSV文件路径
            start_date: 模拟开始时间，格式："YYYY-MM-DD HH:MM"
            sample_ratio: 采样比例

        Returns:
            按时间排序的帖子列表
        """
        print(f"📂 加载帖子数据并按时间排序...")

        # 读取CSV
        df = pd.read_csv(csv_path, encoding='utf-8', dtype=str)

        # 处理时间字段
        df['created_datetime'] = pd.to_datetime(df['created_date'], format='%Y-%m-%d %H:%M')

        # 过滤起始时间之前的数据
        self.start_time = pd.to_datetime(start_date)
        df = df[df['created_datetime'] >= self.start_time]

        # 按时间排序
        df = df.sort_values('created_datetime')

        # 采样（保持时间分布均匀）
        if sample_ratio < 1.0:
            sample_size = int(len(df) * sample_ratio)
            # 等间隔采样以保持时间分布
            indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            df = df.iloc[indices]

        print(f"   原始数据: {len(df)} 条")
        print(f"   时间范围: {df['created_datetime'].min()} 到 {df['created_datetime'].max()}")

        # 转换为字典列表
        posts = []
        for _, row in df.iterrows():
            post = {
                'content': row['content'],
                'platform': row.get('platform', 'unknown'),
                'title': row.get('title', ''),
                # 'original_likes': row.get('like_count', 0),
                # 'original_comments': row.get('comment_count', 0),
                'original_likes': 0,
                'original_comments': 0,
                'created_datetime': row['created_datetime'],
                'stance': row.get('stance', '中立'),
                'sentiment': row.get('sentiment', '中立'),
                # 新增：多媒体URL字段
                'img_urls': row.get('img_urls', ''),
                'video_urls': row.get('video_urls', '')
            }

            # 优先使用现有的post_id，如果没有则生成新的
            if pd.notna(row.get('post_id')) and row.get('post_id'):
                post['post_id'] = row['post_id']
            else:
                # 只有在CSV中没有post_id或为空时才生成新的
                import uuid
                post['post_id'] = f"post_{uuid.uuid4().hex[:6]}"

            posts.append(post)

        self.all_posts = posts
        # 输出前几个看看
        # for p in posts[:3]:
        #     print(f"   示例帖子: {p['post_id']} | {p['created_datetime']} | {p['content'][:30]} | title={p['title']} | platform={p['platform']}")
        self.current_time = self.start_time

        print(f"✅ 准备了 {len(posts)} 个帖子用于时间序列模拟")
        return posts

    def get_available_posts_at_time(self, current_time: datetime) -> List[Dict[str, Any]]:
        """
        获取在指定时间点可以曝光的帖子

        Args:
            current_time: 当前时间点

        Returns:
            可曝光的帖子列表
        """
        available_posts = []
        for post in self.all_posts:
            if post['created_datetime'] <= current_time:
                available_posts.append(post)

        return available_posts

    async def run_time_series_simulation(self,
                                         csv_path: str,
                                         user_path: str,
                                         start_date: str = "2025-03-31 24:00",
                                         sample_ratio: float = 0.4,
                                         max_time_steps: int = 10,
                                         time_step_hours: int = 6,
                                         posts_per_round: int = 3,
                                         users_per_post: int = 6,
                                         config: SimulationConfig = None) -> Dict[str, Any]:
        """
        运行时间序列模拟

        Args:
            csv_path: 帖子数据文件路径
            user_path: 用户数据文件路径
            start_date: 开始时间
            sample_ratio: 采样比例
            max_time_steps: 最大时间步数
            time_step_hours: 时间步长（小时）
            posts_per_round: 每轮帖子数
            users_per_post: 每个帖子的用户数
            config: 模拟配置

        Returns:
            模拟结果
        """
        print(f"🕒 开始时间序列模拟")
        print("=" * 60)

        self.time_step_hours = time_step_hours

        # 加载和准备数据
        posts = self.load_and_prepare_posts(csv_path, start_date, sample_ratio)

        # 初始化模拟环境 - 使用已有用户而非重新生成
        user_manager = UserProfileManager()
        try:
            # 尝试加载已有用户文件
            users_count = user_manager.load_users_from_file(user_path)
            print(f"📋 加载已有用户: {users_count} 个")
        except:
            # 如果没有已有用户文件，则生成新用户
            print("📋 未找到已有用户文件，生成新用户...")
            user_manager.generate_users(count=20, filename=f"timeseries_{self.batch_id}.csv")

        # 使用直接的ContentDistributor
        self.distributor = ContentDistributor(self.batch_id)
        all_users = user_manager.get_all_users()

        print(f"🎯 时间序列模拟参数:")
        print(f"   开始时间: {start_date}")
        print(f"   时间步长: {time_step_hours} 小时")
        print(f"   最大步数: {max_time_steps}")
        print(f"   总用户数: {len(all_users)}")

        # 初始化标志
        distributor_initialized = False

        # 时间序列模拟循环
        for step in range(max_time_steps):
            current_time = self.start_time + timedelta(hours=step * time_step_hours)
            self.current_time = current_time

            print(f"\n🕐 === 时间步 {step + 1}/{max_time_steps}: {current_time.strftime('%Y-%m-%d %H:%M')} ===")

            # 获取当前时间可曝光的帖子
            available_posts = self.get_available_posts_at_time(current_time)

            if len(available_posts) < posts_per_round:
                print(f"⚠️ 可用帖子不足 ({len(available_posts)} < {posts_per_round})，使用全部可用帖子")
                posts_per_round_actual = len(available_posts)
            else:
                posts_per_round_actual = posts_per_round

            if posts_per_round_actual == 0:
                print("⏭️ 无可用帖子，跳过此时间步")
                continue

            print(f"📊 当前时间点数据:")
            print(f"   可用帖子: {len(available_posts)} 个")
            print(f"   本轮使用: {posts_per_round_actual} 个帖子")

            # 初始化分发器（首次有可用帖子时）
            if not distributor_initialized:
                print("🔧 首次初始化分发器...")
                self.distributor.initialize_batch(available_posts, all_users)
                distributor_initialized = True
            else:
                # 更新分发器的可用帖子
                self._update_distributor_posts(available_posts)

            # 调试信息：检查分发器状态
            print(f"   分发器状态检查:")
            if self.distributor.distribution_plan:
                posts_count = len(self.distributor.distribution_plan.get('posts', {}))
                users_count = len(self.distributor.distribution_plan.get('users', {}))
                print(f"      - 可用帖子: {posts_count} 个")
                print(f"      - 可用用户: {users_count} 个")
            else:
                print(f"      - 分发计划未初始化")

            try:
                # 运行单轮模拟
                round_result = await self._run_time_series_round(
                    round_number=step + 1,
                    posts_per_round=posts_per_round_actual,
                    users_per_post=users_per_post,
                    rounds_per_post=1,
                    config=config
                )

                # 保存轮次结果
                self.round_results[step + 1] = round_result

                # 记录本轮参与的用户ID（从分发计划中获取）
                planned_user_ids = self._get_round_active_users(step + 1)
                self.round_active_users[step + 1] = planned_user_ids

                # 收集模拟的立场和情感数据（从实际参与的用户中收集）
                round_result = self.round_results.get(step + 1, {})
                # 确保round_result包含batch_id信息
                round_result['batch_id'] = self.batch_id
                print(f"📊 调试信息: 轮次结果结构 = {list(round_result.keys()) if round_result else '空'}")
                if round_result:
                    print(f"   结果详情: {str(round_result)[:200]}...")
                actual_user_ids = self._get_actual_active_users_from_results(round_result)
                print(f"📊 调试信息: 第{step + 1}轮, 计划用户数={len(planned_user_ids)}, 实际用户数={len(actual_user_ids)}")
                print(f"   实际用户ID: {actual_user_ids[:5]}..." if len(actual_user_ids) > 5 else f"   实际用户ID: {actual_user_ids}")
                sim_stance, sim_sentiment, interaction_count = self._collect_simulation_data_by_users(actual_user_ids)

                # 记录数据（不再记录真实帖子的立场情感，只记录用户的）
                self.time_points.append(current_time)
                self.stance_history.append({
                    'time': current_time,
                    'sim_stance': sim_stance,
                    'sim_sentiment': sim_sentiment,
                    'interaction_count': interaction_count,
                    'available_posts': len(available_posts),
                    'active_users': actual_user_ids
                })

                print(f"📈 模拟结果:")
                print(f"   模拟立场均值: {sim_stance:.3f}")
                print(f"   模拟情感均值: {sim_sentiment:.3f}")
                print(f"   参与用户数: {interaction_count}")

            except Exception as e:
                print(f"❌ 第 {step + 1} 轮模拟失败: {e}")
                continue

        # 保存和可视化结果
        self._save_results()
        self._plot_comparison()

        return {
            'batch_id': self.batch_id,
            'time_points': [t.isoformat() for t in self.time_points],
            'stance_history': self.stance_history
        }

    async def _run_time_series_round(self, round_number: int,
                                     posts_per_round: int = 5,
                                     users_per_post: int = 10,
                                     rounds_per_post: int = 1,
                                     config: SimulationConfig = None) -> Dict[str, Any]:
        """
        运行时间序列单轮模拟

        Args:
            round_number: 轮次号
            posts_per_round: 每轮参与的帖子数量
            users_per_post: 每个帖子分配的用户数量
            rounds_per_post: 每个帖子内的交互轮数
            config: 模拟配置

        Returns:
            轮次结果
        """
        print(f"🎮 === 开始第 {round_number} 轮时间序列模拟 ===")

        # 生成分发计划
        distribution = self.distributor.generate_round_distribution(
            round_number=round_number,
            posts_per_round=posts_per_round,
            users_per_post=users_per_post,
            hot_post_ratio=0.2 if round_number > 1 else 0.0  # 时间序列模拟中热门帖子比例较低
        )

        if config is None:
            config = SimulationConfig(
                max_concurrent_requests=3,
                action_probability=0.8,
                comment_probability=0.6,
                export_prompts=False,  # 时间序列模拟默认不导出prompt
                prompt_export_dir=f"Output/prompt_exports/timeseries_{self.batch_id}/round_{round_number}"
            )

        if config.prompt_export_dir is None and config.export_prompts:
            config.prompt_export_dir = f"Output/prompt_exports/timeseries_{self.batch_id}/round_{round_number}"

        # 创建引擎
        engine = SimulationEngine(config)

        # 轮次结果统计
        round_results = {
            'round_number': round_number,
            'batch_id': self.batch_id,  # 确保包含batch_id
            'posts': [],
            'total_actions': 0,
            'start_time': datetime.now().isoformat()
        }

        try:
            # 为每个分发的帖子进行模拟
            for post_idx, (post_id, post_dist) in enumerate(distribution['posts'].items(), 1):
                print(f"📝 模拟帖子 {post_idx}/{len(distribution['posts'])}: {post_id}")

                # 获取帖子内容
                post_data = self.distributor.distribution_plan['posts'][post_id]
                post_content = post_data['content']

                # 获取分配的用户
                assigned_user_ids = post_dist['assigned_users']
                user_profiles = []
                for user_id in assigned_user_ids:
                    user_data = self.distributor.distribution_plan['users'][user_id]
                    user_profiles.append(user_data['profile'])

                print(f"   分配用户: {len(user_profiles)} 个")

                # 获取多媒体URL信息（如果存在）
                img_urls = post_data.get('img_urls', '')
                video_urls = post_data.get('video_urls', '')

                # 创建会话（支持多模态分析）
                session_post_id = await engine.create_session_with_multimodal(
                    post_content=post_content,
                    post_id=post_id,
                    batch_id=self.batch_id,
                    img_urls=img_urls,
                    video_urls=video_urls
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

                # 记录帖子结果
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

        print(f"✅ 第 {round_number} 轮时间序列模拟完成")
        print(f"   参与帖子: {len(round_results['posts'])} 个")
        print(f"   总行为数: {round_results['total_actions']} 个")
        print(f"   耗时: {round_results['duration_minutes']:.1f} 分钟")

        return round_results

    def _update_distributor_posts(self, available_posts: List[Dict[str, Any]]):
        """更新分发器的可用帖子（简化版本）"""
        import uuid

        # 检查并确保每个帖子都有post_id
        for post in available_posts:
            if 'post_id' not in post or not post['post_id']:
                # 只有在没有post_id或为空时才生成新的
                post['post_id'] = f"post_{uuid.uuid4().hex[:6]}"
                print(f"   ⚠️ 为帖子生成新post_id: {post['post_id']}")
            # else:
            #     print(f"   ✅ 使用现有post_id: {post['post_id']}")

        # 创建新的帖子字典，保持原有结构
        new_posts_dict = {}
        for post in available_posts:
            post_id = post['post_id']

            # 如果帖子已存在，保留原有数据，否则创建新数据
            if (self.distributor.distribution_plan and
                'posts' in self.distributor.distribution_plan and
                    post_id in self.distributor.distribution_plan['posts']):
                # 保留已有帖子数据
                new_posts_dict[post_id] = self.distributor.distribution_plan['posts'][post_id]
            else:
                # 创建新帖子数据
                new_posts_dict[post_id] = {
                    'post_id': post_id,
                    'content': post['content'],
                    'platform': post.get('platform', 'unknown'),
                    'title': post.get('title', ''),
                    # 新增：多媒体URL信息
                    'img_urls': post.get('img_urls', ''),
                    'video_urls': post.get('video_urls', ''),
                    'original_metrics': {
                        'likes': post.get('original_likes', 0),
                        'comments': post.get('original_comments', 0)
                    },
                    'simulation_metrics': {
                        'total_interactions': 0,
                        'total_likes': 0,
                        'total_comments': 0,
                        'unique_users': 0,
                        'heat_score': 0.0
                    },
                    'round_history': []
                }

        # 更新分发计划中的帖子
        if self.distributor.distribution_plan:
            self.distributor.distribution_plan['posts'] = new_posts_dict
            # 保存更新后的计划
            self.distributor._save_plan()

    def _get_round_active_users(self, round_number: int) -> List[str]:
        """
        获取本轮参与的用户ID列表

        Args:
            round_number: 轮次号

        Returns:
            参与用户ID列表
        """
        active_users = set()

        # 从分发计划中获取本轮分配的用户
        if (self.distributor.distribution_plan and
            'rounds' in self.distributor.distribution_plan and
                len(self.distributor.distribution_plan['rounds']) >= round_number):

            round_data = self.distributor.distribution_plan['rounds'][round_number - 1]
            for post_id, post_data in round_data['posts'].items():
                assigned_users = post_data.get('assigned_users', [])
                active_users.update(assigned_users)

        return list(active_users)

    def _get_actual_active_users_from_results(self, round_result: Dict) -> List[str]:
        """
        从模拟结果中获取实际产生行为的用户ID列表

        Args:
            round_result: 模拟轮次结果字典

        Returns:
            实际参与用户ID列表
        """
        active_users = set()

        # 从batch_id中读取distribution_plan.json文件
        batch_id = round_result.get('batch_id')
        if not batch_id:
            print("[DEBUG] 没有找到batch_id")
            return []

        # 构建导出目录路径和distribution_plan文件路径
        export_dir = os.path.join(self.output_dir, 'exports', batch_id)
        distribution_plan_file = os.path.join(export_dir, 'distribution_plan.json')

        print(f"[DEBUG] 查找分配计划文件: {distribution_plan_file}")

        if not os.path.exists(distribution_plan_file):
            print(f"[DEBUG] 分配计划文件不存在: {distribution_plan_file}")
            return []

        try:
            # 读取分配计划
            with open(distribution_plan_file, 'r', encoding='utf-8') as f:
                distribution_plan = json.load(f)

            # 获取当前轮次号
            current_round = round_result.get('round_number', 0)
            print(f"[DEBUG] 当前轮次: {current_round}")

            # 从分配计划中找到对应轮次的用户分配
            for round_info in distribution_plan.get('rounds', []):
                if round_info.get('round_number') == current_round:
                    posts = round_info.get('posts', {})
                    for post_id, post_info in posts.items():
                        assigned_users = post_info.get('assigned_users', [])
                        active_users.update(assigned_users)
                        print(f"[DEBUG] 帖子{post_id}分配了{len(assigned_users)}个用户")
                    break

            print(f"[DEBUG] 总计分配用户数={len(active_users)}")
            # print(f"[DEBUG] 分配用户ID: {list(active_users)}")

        except Exception as e:
            print(f"[DEBUG] 读取分配计划失败: {e}")
            return []

        return list(active_users)

    def _collect_simulation_data_by_users(self, active_user_ids: List[str]) -> Tuple[float, float, int]:
        """
        基于指定用户ID列表收集立场和情感数据

        Args:
            active_user_ids: 参与用户ID列表

        Returns:
            (平均立场, 平均情感, 参与用户数)
        """
        if not active_user_ids:
            return 0.0, 0.0, 0

        # 获取用户记忆管理器
        try:
            from UserAgent.user_memory_manager import UserMemoryManager
            memory_manager = UserMemoryManager(memory_dir="UserAgent/user_memories", batch_id=self.batch_id)
        except Exception as e:
            print(f"⚠️ 无法创建记忆管理器: {e}")
            return 0.0, 0.0, 0

        # 收集指定用户的当前立场和情感
        stance_values = []
        sentiment_values = []
        valid_users = 0

        print(f"🔍 正在收集 {len(active_user_ids)} 个用户的立场情感数据...")

        for user_id in active_user_ids:
            try:
                user_memory = memory_manager._get_user_memory(user_id)
                if user_memory:
                    # 获取用户当前的立场和情感（最新值）
                    current_stance = user_memory.current_stance_value
                    current_sentiment = user_memory.current_sentiment_value

                    stance_values.append(current_stance)
                    sentiment_values.append(current_sentiment)
                    valid_users += 1

                    # print(f"   用户 {user_id}: 立场={current_stance:.3f}, 情感={current_sentiment:.3f}")
                else:
                    print(f"   用户 {user_id}: 无记忆数据")
            except Exception as e:
                print(f"   用户 {user_id}: 获取数据失败 - {e}")

        avg_stance = np.mean(stance_values) if stance_values else 0.0
        avg_sentiment = np.mean(sentiment_values) if sentiment_values else 0.0

        print(f"✅ 数据收集完成: {valid_users} 个有效用户, 平均立场={avg_stance:.3f}, 平均情感={avg_sentiment:.3f}")

        return avg_stance, avg_sentiment, valid_users

    def _save_results(self):
        """保存结果到文件"""
        results_dir = Path(f"Output/timeseries/{self.batch_id}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 保存详细数据
        results_file = results_dir / "timeseries_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'batch_id': self.batch_id,
                'start_time': self.start_time.isoformat(),
                'time_step_hours': self.time_step_hours,
                'stance_history': self.stance_history,
            }, f, indent=2, ensure_ascii=False, default=str)

        print(f"📁 结果已保存到: {results_file}")

    def _plot_comparison(self):
        """绘制用户立场情感变化图"""
        if not self.stance_history:
            print("⚠️ 没有数据可供绘制")
            return

        # 准备数据 - 只需要用户模拟数据
        times = [item['time'] for item in self.stance_history]
        sim_stance = [item['sim_stance'] for item in self.stance_history]
        sim_sentiment = [item['sim_sentiment'] for item in self.stance_history]
        interaction_counts = [item['interaction_count'] for item in self.stance_history]

        # 保存绘图数据点到CSV文件
        self._save_plot_data(times, sim_stance, sim_sentiment, interaction_counts)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'时间序列用户立场情感变化分析 - {self.batch_id}', fontsize=16, fontweight='bold')

        # 用户立场变化
        axes[0, 0].plot(times, sim_stance, 'r-o', label='用户立场', linewidth=2.5, markersize=6, color='#2E86AB')
        axes[0, 0].set_title('用户立场时间变化', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('立场值', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='中立线')
        axes[0, 0].set_ylim(-1.1, 1.1)

        # 用户情感变化
        axes[0, 1].plot(times, sim_sentiment, 's-', label='用户情感', linewidth=2.5, markersize=6, color='#A23B72')
        axes[0, 1].set_title('用户情感时间变化', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('情感值', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='中立线')
        axes[0, 1].set_ylim(-1.1, 1.1)

        # 参与用户数变化
        axes[1, 0].bar(range(len(times)), interaction_counts, alpha=0.7, color='orange')
        axes[1, 0].set_title('每轮参与用户数', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('参与用户数', fontsize=12)
        axes[1, 0].set_xlabel('时间步', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        # 立场情感关系散点图
        axes[1, 1].scatter(sim_stance, sim_sentiment, alpha=0.7, s=80, c=range(len(sim_stance)),
                           cmap='viridis', label='时间序列点')

        # 添加颜色条
        scatter = axes[1, 1].scatter(sim_stance, sim_sentiment, alpha=0.7, s=80, c=range(len(sim_stance)), cmap='viridis')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('时间步', fontsize=10)

        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('立场 vs 情感分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('立场值', fontsize=12)
        axes[1, 1].set_ylabel('情感值', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(-1.1, 1.1)
        axes[1, 1].set_ylim(-1.1, 1.1)

        # 格式化时间轴
        for ax in [axes[0, 0], axes[0, 1]]:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # 保存图表
        results_dir = Path(f"Output/timeseries/{self.batch_id}")
        results_dir.mkdir(parents=True, exist_ok=True)

        plot_file = results_dir / "user_stance_sentiment_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 用户立场情感变化图已保存到: {plot_file}")

        # 打印统计信息
        self._print_user_statistics()

    def _save_plot_data(self, times: List[datetime], sim_stance: List[float],
                        sim_sentiment: List[float], interaction_counts: List[int]):
        """
        保存绘图数据点到CSV文件

        Args:
            times: 时间点列表
            sim_stance: 立场数值列表
            sim_sentiment: 情感数值列表
            interaction_counts: 参与用户数列表
        """
        results_dir = Path(f"Output/timeseries/{self.batch_id}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 创建包含所有绘图数据的DataFrame
        plot_data = []
        for i, time_point in enumerate(times):
            plot_data.append({
                'time_step': i + 1,
                'datetime': time_point.strftime('%Y-%m-%d %H:%M:%S'),
                'user_stance': sim_stance[i],
                'user_sentiment': sim_sentiment[i],
                'participant_count': interaction_counts[i],
                'batch_id': self.batch_id
            })

        # 保存为CSV
        plot_data_file = results_dir / "plot_data_points.csv"
        df = pd.DataFrame(plot_data)
        df.to_csv(plot_data_file, index=False, encoding='utf-8-sig')

        print(f"📈 绘图数据点已保存到: {plot_data_file}")

        # 保存为JSON格式（更详细的数据）
        detailed_data = {
            'batch_id': self.batch_id,
            'time_step_hours': self.time_step_hours,
            'total_time_steps': len(times),
            'data_points': plot_data,
            'summary': {
                'stance_avg': float(np.mean(sim_stance)) if sim_stance else 0.0,
                'stance_min': float(min(sim_stance)) if sim_stance else 0.0,
                'stance_max': float(max(sim_stance)) if sim_stance else 0.0,
                'sentiment_avg': float(np.mean(sim_sentiment)) if sim_sentiment else 0.0,
                'sentiment_min': float(min(sim_sentiment)) if sim_sentiment else 0.0,
                'sentiment_max': float(max(sim_sentiment)) if sim_sentiment else 0.0,
                'participant_avg': float(np.mean(interaction_counts)) if interaction_counts else 0.0,
                'participant_total': sum(interaction_counts)
            }
        }

        plot_data_json = results_dir / "plot_data_points.json"
        with open(plot_data_json, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)

        print(f"📊 详细绘图数据已保存到: {plot_data_json}")

    def _print_user_statistics(self):
        """打印用户统计信息"""
        if not self.stance_history:
            return

        print(f"\n📊 用户立场情感统计摘要:")
        print(f"时间序列模拟批次: {self.batch_id}")
        print(f"时间间隔: 每{self.time_step_hours}小时")
        print(f"总时间步数: {len(self.stance_history)} 个")

        # 收集所有立场和情感数据
        all_stances = [item['sim_stance'] for item in self.stance_history]
        all_sentiments = [item['sim_sentiment'] for item in self.stance_history]
        all_interactions = [item['interaction_count'] for item in self.stance_history]

        # 立场统计
        avg_stance = np.mean(all_stances)
        print(f"\n用户立场分析:")
        print(f"  整体平均立场: {avg_stance:.3f}")
        print(f"  立场范围: {min(all_stances):.3f} 到 {max(all_stances):.3f}")
        if avg_stance > 0.1:
            print(f"  总体倾向: 支持")
        elif avg_stance < -0.1:
            print(f"  总体倾向: 反对")
        else:
            print(f"  总体倾向: 中立")

        # 情感统计
        avg_sentiment = np.mean(all_sentiments)
        print(f"\n用户情感分析:")
        print(f"  整体平均情感: {avg_sentiment:.3f}")
        print(f"  情感范围: {min(all_sentiments):.3f} 到 {max(all_sentiments):.3f}")
        if avg_sentiment > 0.1:
            print(f"  总体倾向: 积极")
        elif avg_sentiment < -0.1:
            print(f"  总体倾向: 消极")
        else:
            print(f"  总体倾向: 中立")

        # 参与度统计
        print(f"\n参与度分析:")
        print(f"  平均参与用户数: {np.mean(all_interactions):.1f}")
        print(f"  参与度范围: {min(all_interactions)} 到 {max(all_interactions)} 人")
        print(f"  总参与人次: {sum(all_interactions)}")

        # 获取所有参与过的用户
        all_active_users = set()
        for round_users in self.round_active_users.values():
            all_active_users.update(round_users)
        print(f"  累计参与用户数: {len(all_active_users)} 人")


async def main():
    """主函数"""
    print("🕒 时间序列社交媒体模拟测试")
    print("=" * 60)

    # 创建时间序列模拟实例
    ts_sim = TimeSeriesSimulation()

    # 配置支持多模态分析的设置
    config = SimulationConfig(
        max_concurrent_requests=10,
        request_timeout=60,
        model_name="qwen-max",
        action_probability=0.8,
        comment_probability=0.5,
        export_prompts=False,
        # 启用多模态分析功能
        enable_multimodal=False,
        multimodal_model="qwen-vl-max-2025-08-13",
        multimodal_max_images=8,
        multimodal_timeout=60,
        multimodal_fallback_on_error=True,
        # 启用多模态缓存功能
        multimodal_use_cache=True,
        multimodal_cache_dir=None,  # 使用默认缓存目录
        multimodal_cache_filename=None  # 使用默认缓存文件名
    )

    # 运行时间序列模拟
    results = await ts_sim.run_time_series_simulation(
        csv_path="Data/integrated_data/XMSU7D_integrated_articles.csv",
        user_path="demo_users_0907_2.csv",  # 使用已有用户文件
        start_date="2025-03-31 18:00",  # 从2025年3月31日18:00开始
        sample_ratio=0.8,               # 采样比例
        max_time_steps=10,               # 运行10个时间步
        time_step_hours=6,              # 每6小时一步
        posts_per_round=3,              # 每轮3个帖子（减少以便观察多模态效果）
        users_per_post=10,               # 每个帖子20个用户
        config=config
    )

    print("\n🎉 时间序列模拟完成！")
    print(f"📊 结果保存在: Output/timeseries/{ts_sim.batch_id}/")


if __name__ == "__main__":
    # 选择运行版本：
    asyncio.run(main())  # 带多模态分析
