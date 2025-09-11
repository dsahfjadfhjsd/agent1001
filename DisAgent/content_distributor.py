# -*- coding: utf-8 -*-
"""
内容分发算法模块

负责在多轮模拟中智能分配帖子和用户，模拟真实社交媒体的内容分发机制
"""

import random
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import uuid


class ContentDistributor:
    """内容分发算法类"""

    def __init__(self, batch_id: str, batch_dir: str = None):
        """
        初始化内容分发器

        Args:
            batch_id: 批次ID
            batch_dir: 批次目录路径
        """
        self.batch_id = batch_id

        if batch_dir is None:
            batch_dir = f"Output/exports/{batch_id}"

        self.batch_dir = Path(batch_dir)
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        # 分发计划文件
        self.distribution_plan_file = self.batch_dir / "distribution_plan.json"
        self.round_results_file = self.batch_dir / "round_results.json"

        # 当前分发计划
        self.distribution_plan = self._load_or_create_plan()

        print(f"🎯 内容分发器初始化完成 - 批次: {batch_id}")

    def initialize_batch(self, posts: List[Dict[str, Any]], users: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        初始化批次，为所有帖子生成post_id并创建初始分发计划

        Args:
            posts: 帖子列表
            users: 用户列表

        Returns:
            初始化后的分发计划
        """
        print(f"📋 初始化批次分发计划...")

        # 为每个帖子检查或生成post_id
        for i, post in enumerate(posts):
            if 'post_id' not in post or not post['post_id']:
                # 只有在没有post_id或为空时才生成新的ID
                post['post_id'] = f"post_{uuid.uuid4().hex[:6]}"
                print(f"   🆔 为帖子生成新ID: {post['post_id']}")
            else:
                print(f"   ✅ 使用现有post_id: {post['post_id']}")

        # 初始化分发计划
        self.distribution_plan = {
            'batch_id': self.batch_id,
            'created_at': datetime.now().isoformat(),
            'posts': {post['post_id']: {
                'post_id': post['post_id'],
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
            } for post in posts},
            'users': {user['user_id']: {
                'user_id': user['user_id'],
                'profile': user,
                'interaction_history': [],
                'activity_score': 0.0,
                'last_active_round': 0
            } for user in users},
            'rounds': []
        }

        self._save_plan()
        print(f"✅ 批次初始化完成 - {len(posts)} 个帖子，{len(users)} 个用户")
        return self.distribution_plan

    def generate_round_distribution(self, round_number: int,
                                    posts_per_round: int = 5,
                                    users_per_post: int = 10,
                                    hot_post_ratio: float = 0.4) -> Dict[str, Any]:
        """
        生成指定轮次的分发计划

        Args:
            round_number: 轮次号
            posts_per_round: 每轮参与的帖子数量
            users_per_post: 每个帖子分配的用户数量
            hot_post_ratio: 热门帖子比例（从之前轮次选择）

        Returns:
            当前轮次的分发计划
        """
        print(f"\n🎲 生成第 {round_number} 轮分发计划...")

        if round_number == 1:
            # 第一轮：完全随机选择
            selected_posts = self._select_first_round_posts(posts_per_round)
        else:
            # 后续轮次：结合热门度和随机选择
            selected_posts = self._select_subsequent_round_posts(
                round_number, posts_per_round, hot_post_ratio
            )

        # 为每个选中的帖子分配用户
        round_distribution = {
            'round_number': round_number,
            'created_at': datetime.now().isoformat(),
            'posts': {},
            'total_posts': len(selected_posts),
            'total_planned_interactions': 0
        }

        for post_id in selected_posts:
            # 选择用户
            assigned_users = self._select_users_for_post(post_id, round_number, users_per_post)

            round_distribution['posts'][post_id] = {
                'post_id': post_id,
                'assigned_users': assigned_users,
                'user_count': len(assigned_users),
                'selection_reason': self._get_post_selection_reason(post_id, round_number)
            }

            round_distribution['total_planned_interactions'] += len(assigned_users)

            print(f"   📝 {post_id}: {len(assigned_users)} 个用户")

        # 保存到分发计划中
        self.distribution_plan['rounds'].append(round_distribution)
        self._save_plan()

        print(f"✅ 第 {round_number} 轮分发计划完成")
        print(f"   选中帖子: {len(selected_posts)} 个")
        print(f"   计划交互: {round_distribution['total_planned_interactions']} 次")

        return round_distribution

    def update_round_results(self, round_number: int, results: Dict[str, Any]):
        """
        更新轮次结果，用于后续轮次的热门度计算

        Args:
            round_number: 轮次号
            results: 轮次结果数据
        """
        print(f"📊 更新第 {round_number} 轮结果...")

        # 更新帖子指标
        for post_result in results.get('posts', []):
            post_id = post_result['post_id']
            if post_id in self.distribution_plan['posts']:
                post_data = self.distribution_plan['posts'][post_id]

                # 更新累计指标
                post_data['simulation_metrics']['total_interactions'] += post_result['actions_count']
                post_data['simulation_metrics']['unique_users'] += post_result['users_count']

                # 计算热门度分数（交互数量 * 权重 + 用户数量 * 权重）
                interactions_weight = 1.0
                users_weight = 2.0
                post_data['simulation_metrics']['heat_score'] = (
                    post_data['simulation_metrics']['total_interactions'] * interactions_weight +
                    post_data['simulation_metrics']['unique_users'] * users_weight
                )

                # 记录轮次历史
                post_data['round_history'].append({
                    'round': round_number,
                    'interactions': post_result['actions_count'],
                    'users': post_result['users_count'],
                    'action_types': post_result.get('action_types', {})
                })

        # 更新用户活跃度
        self._update_user_activity(round_number, results)

        self._save_plan()
        print(f"✅ 第 {round_number} 轮结果更新完成")

    def get_hot_posts(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        获取热门帖子列表

        Args:
            top_k: 返回前k个热门帖子

        Returns:
            热门帖子列表
        """
        posts_with_heat = []
        for post_id, post_data in self.distribution_plan['posts'].items():
            if post_data['simulation_metrics']['heat_score'] > 0:
                posts_with_heat.append({
                    'post_id': post_id,
                    'heat_score': post_data['simulation_metrics']['heat_score'],
                    'total_interactions': post_data['simulation_metrics']['total_interactions'],
                    'unique_users': post_data['simulation_metrics']['unique_users'],
                    'content_preview': post_data['content'][:50] + "..."
                })

        # 按热门度排序
        posts_with_heat.sort(key=lambda x: x['heat_score'], reverse=True)
        return posts_with_heat[:top_k]

    def get_unused_posts(self) -> List[str]:
        """
        获取尚未参与过模拟的帖子ID列表

        Returns:
            未使用的帖子ID列表
        """
        unused_posts = []
        for post_id, post_data in self.distribution_plan['posts'].items():
            if not post_data['round_history']:
                unused_posts.append(post_id)
        return unused_posts

    def get_distribution_summary(self) -> Dict[str, Any]:
        """
        获取分发计划摘要

        Returns:
            分发摘要信息
        """
        total_posts = len(self.distribution_plan['posts'])
        used_posts = len([p for p in self.distribution_plan['posts'].values() if p['round_history']])
        total_users = len(self.distribution_plan['users'])
        total_rounds = len(self.distribution_plan['rounds'])

        summary = {
            'batch_id': self.batch_id,
            'total_posts': total_posts,
            'used_posts': used_posts,
            'unused_posts': total_posts - used_posts,
            'total_users': total_users,
            'total_rounds': total_rounds,
            'hot_posts': self.get_hot_posts(3),
            'round_summaries': []
        }

        for round_data in self.distribution_plan['rounds']:
            round_summary = {
                'round': round_data['round_number'],
                'posts_count': round_data['total_posts'],
                'planned_interactions': round_data['total_planned_interactions']
            }
            summary['round_summaries'].append(round_summary)

        return summary

    def _load_or_create_plan(self) -> Dict[str, Any]:
        """加载或创建分发计划"""
        if self.distribution_plan_file.exists():
            with open(self.distribution_plan_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}

    def _save_plan(self):
        """保存分发计划（只保存rounds信息）"""
        # 只保存rounds数组，不保存其他冗余信息
        simplified_plan = {
            'batch_id': self.distribution_plan['batch_id'],
            'created_at': self.distribution_plan['created_at'],
            'rounds': self.distribution_plan['rounds']
        }

        with open(self.distribution_plan_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_plan, f, indent=2, ensure_ascii=False)

    def _select_first_round_posts(self, count: int) -> List[str]:
        """选择第一轮帖子（完全随机）"""
        all_post_ids = list(self.distribution_plan['posts'].keys())
        return random.sample(all_post_ids, min(count, len(all_post_ids)))

    def _select_subsequent_round_posts(self, round_number: int, count: int, hot_ratio: float) -> List[str]:
        """选择后续轮次帖子（结合热门度和随机，增加随机性避免固化）"""

        # 随机策略选择概率
        # 30% 概率：完全随机选择（避免热门帖子固化）
        # 60% 概率：热门帖子 + 随机帖子混合
        # 10% 概率：大部分热门帖子 + 少量随机帖子

        strategy_choice = random.random()

        if strategy_choice < 0.3:
            # 策略1：完全随机选择
            print(f"   📦 策略：完全随机选择")
            all_post_ids = list(self.distribution_plan['posts'].keys())
            return random.sample(all_post_ids, min(count, len(all_post_ids)))

        elif strategy_choice < 0.9:
            # 策略2：热门帖子 + 随机帖子混合（使用原始hot_ratio）
            print(f"   📦 策略：热门+随机混合 (热门比例: {hot_ratio:.1f})")
            return self._select_mixed_posts(count, hot_ratio)

        else:
            # 策略3：大部分热门帖子 + 少量随机帖子
            enhanced_hot_ratio = min(0.8, hot_ratio + 0.3)  # 提高热门比例到最高80%
            print(f"   📦 策略：偏向热门选择 (热门比例: {enhanced_hot_ratio:.1f})")
            return self._select_mixed_posts(count, enhanced_hot_ratio)

    def _select_mixed_posts(self, count: int, hot_ratio: float) -> List[str]:
        """选择混合帖子（热门+随机）"""
        # 计算热门帖子和随机帖子的数量
        hot_count = int(count * hot_ratio)
        random_count = count - hot_count

        selected_posts = []

        # 选择热门帖子
        if hot_count > 0:
            hot_posts = self.get_hot_posts(hot_count * 2)  # 获取更多候选
            if hot_posts:
                # 从热门帖子中再加一点随机性，不总是选择最热门的
                available_hot = [p['post_id'] for p in hot_posts]
                selected_hot = random.sample(available_hot, min(hot_count, len(available_hot)))
                selected_posts.extend(selected_hot)
                print(f"     🔥 选中 {len(selected_hot)} 个热门帖子")

        # 选择随机帖子（优先选择未使用的）
        if random_count > 0:
            unused_posts = self.get_unused_posts()

            # 70%概率优先选择未使用的帖子，30%概率从所有帖子中选择
            prefer_unused = random.random() < 0.7

            if prefer_unused and len(unused_posts) >= random_count:
                random_posts = random.sample(unused_posts, random_count)
                print(f"     🎲 选中 {len(random_posts)} 个新帖子")
            else:
                # 从所有帖子中随机选择（排除已选中的）
                all_posts = list(self.distribution_plan['posts'].keys())
                available_posts = [p for p in all_posts if p not in selected_posts]
                if available_posts:
                    actual_random_count = min(random_count, len(available_posts))
                    random_posts = random.sample(available_posts, actual_random_count)
                    print(f"     🎲 选中 {len(random_posts)} 个随机帖子")
                else:
                    random_posts = []

            selected_posts.extend(random_posts)

        return selected_posts

    def _select_users_for_post(self, post_id: str, round_number: int, count: int) -> List[str]:
        """为帖子选择用户"""
        all_users = list(self.distribution_plan['users'].keys())

        if round_number == 1:
            # 第一轮完全随机
            return random.sample(all_users, min(count, len(all_users)))
        else:
            # 后续轮次考虑用户活跃度和新用户比例
            active_users = [uid for uid, udata in self.distribution_plan['users'].items()
                            if udata['last_active_round'] == round_number - 1]
            inactive_users = [uid for uid, udata in self.distribution_plan['users'].items()
                              if udata['last_active_round'] < round_number - 1]

            # 20%活跃用户，80%非活跃用户
            active_count = min(int(count * 0.2), len(active_users))
            inactive_count = count - active_count

            selected = []
            if active_count > 0 and active_users:
                selected.extend(random.sample(active_users, min(active_count, len(active_users))))
            if inactive_count > 0 and inactive_users:
                selected.extend(random.sample(inactive_users, min(inactive_count, len(inactive_users))))

            # 如果不够，从所有用户中补充
            if len(selected) < count:
                remaining_users = [u for u in all_users if u not in selected]
                if remaining_users:
                    additional = random.sample(remaining_users, min(count - len(selected), len(remaining_users)))
                    selected.extend(additional)

            return selected

    def _get_post_selection_reason(self, post_id: str, round_number: int) -> str:
        """获取帖子选择原因"""
        if round_number == 1:
            return "首轮随机选择"

        post_data = self.distribution_plan['posts'][post_id]
        if post_data['round_history']:
            return f"热门帖子 (热度: {post_data['simulation_metrics']['heat_score']:.1f})"
        else:
            return "新帖子随机选择"

    def _update_user_activity(self, round_number: int, results: Dict[str, Any]):
        """更新用户活跃度"""
        # 这里可以根据实际的用户行为数据更新活跃度
        # 简化版本：标记参与了本轮的用户
        for post_result in results.get('posts', []):
            post_id = post_result['post_id']
            if round_number <= len(self.distribution_plan['rounds']):
                round_data = self.distribution_plan['rounds'][round_number - 1]
                if post_id in round_data['posts']:
                    assigned_users = round_data['posts'][post_id]['assigned_users']
                    for user_id in assigned_users:
                        if user_id in self.distribution_plan['users']:
                            self.distribution_plan['users'][user_id]['last_active_round'] = round_number
                            self.distribution_plan['users'][user_id]['activity_score'] += 1.0
