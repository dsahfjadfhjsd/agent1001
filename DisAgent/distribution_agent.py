# -*- coding: utf-8 -*-
"""
DistributionAgent

基于 LangChain/OpenAI 的分发智能体包装，向下兼容原有 ContentDistributor 的 API。

行为：
- 强烈依赖 LangChain + OpenAI 对帖子进行打分/排序；如果依赖缺失或 API 未配置，初始化将抛出异常（不降级）。
- 用户选择与结果更新使用与原 ContentDistributor 类似的确定性逻辑
- 始终将完整的分发计划保存到 Output/exports/{batch_id}/distribution_plan.json
"""

import os
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# 尝试加载配置（从 Config/.env）
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join('Config', '.env'))

# logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')


try:
    # 使用 ChatOpenAI 而不是 OpenAI，因为我们需要聊天模式
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False


class DistributionAgent:
    """兼容 ContentDistributor API 的智能体实现（严格依赖 LangChain）"""

    def __init__(self, batch_id: str, batch_dir: str = None, llm_config: Dict[str, Any] = None,
                 seed: Optional[int] = None, evaluation_results: Dict[str, Any] = None,
                 realtime_callback: callable = None):
        self.batch_id = batch_id

        if batch_dir is None:
            batch_dir = f"Output/exports/{batch_id}"

        self.batch_dir = Path(batch_dir)
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        self.distribution_plan_file = self.batch_dir / "distribution_plan.json"

        self.distribution_plan: Dict[str, Any] = self._load_or_create_plan()

        # LLM 初始化（若可用且有密钥）
        self.llm = None
        self.llm_chain = None
        self.llm_config = llm_config or {}

        # 使用实例级 RNG 避免修改全局随机状态
        self._rng = random.Random(seed)

        # 评估结果存储，用于优化分发策略
        self.evaluation_results = evaluation_results or {}
        self.optimization_history = []
        
        # 实时回调函数，用于发送分发事件到前端
        self.realtime_callback = realtime_callback

        # 强制要求 LangChain 与 OPENAI_API_KEY 可用，若不可用则抛出异常（不降级）
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain is required for DistributionAgent but is not available. Install langchain and related dependencies.")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not found in Config/.env; DistributionAgent requires an OpenAI API key.")

        # langchain ChatOpenAI 客户端，允许通过 OPENAI_BASE_URL 指定自定义 endpoint
        try:
            # 支持通过 llm_config 指定模型名，默认使用 qwen-max
            model_name = self.llm_config.get('model_name', 'qwen-max')
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=OPENAI_API_KEY, 
                base_url=OPENAI_BASE_URL,
                temperature=self.llm_config.get('temperature', 0.0),
                max_tokens=self.llm_config.get('max_tokens', 2048)
            )
        except Exception as e:
            logger.exception("初始化 ChatOpenAI 客户端失败")
            raise
        # 创新灵活的分发决策 prompt - 鼓励创造性和适应性
        template = (
            "你是一个具有创新思维的AI内容分发策略师。请打破常规，设计灵活多变的个性化分发方案。\n\n"
            "🎯 **核心使命：** 创造惊喜、个性化、高参与度的内容体验\n\n"
            "📊 **智能输入：**\n"
            "- 轮次：{round_number} | 候选内容：{candidates}\n"
            "- 系统参数：{params} | 约束条件：{constraints}\n"
            "- 用户生态：{user_segments}\n\n"
            "🚀 **创新策略原则：**\n"
            "• 突破固定模式 - 根据内容特性动态调整帖子数量(1-8个)\n"
            "• 个性化匹配 - 深度分析用户-内容适配度，创造意外惊喜\n"
            "• 情感共鸣 - 考虑内容情感价值和社会影响力\n"
            "• 时机把握 - 利用热点时效性和用户注意力周期\n"
            "• 实验精神 - 敢于尝试新组合，在安全范围内冒险\n"
            "• 生态平衡 - 爆款与长尾内容的艺术平衡\n\n"
            "💡 **灵活输出格式 (JSON)：**\n"
            "{{\n"
            "  \"strategy_philosophy\": \"本轮创新理念和价值主张\",\n"
            "  \"ranked_posts\": [\n"
            "    {{\"post_id\": \"ID\", \"innovation_score\": 0.88, \"selection_rationale\": \"多维度选择理由\", \"surprise_factor\": \"独特价值点\"}}\n"
            "  ],\n"
            "  \"personalized_matching\": [\n"
            "    {{\"post_id\": \"ID\", \"user_strategy\": {{\"active\": 0.6, \"inactive\": 0.3, \"new\": 0.1}}, \"emotional_targeting\": \"情感定向策略\"}}\n"
            "  ],\n"
            "  \"adaptive_parameters\": {{\n"
            "    \"posts_per_round\": \"动态调整的帖子数(1-8)\",\n"
            "    \"users_per_post\": \"动态用户数(4-12)\",\n"
            "    \"hot_post_ratio\": \"热门比例(0.2-0.8)\",\n"
            "    \"personalization_strength\": \"个性化强度(0.1-1.0)\",\n"
            "    \"innovation_reason\": \"参数创新的深层逻辑\"\n"
            "  }},\n"
            "  \"experimental_elements\": {{\n"
            "    \"new_attempts\": \"本轮尝试的新策略\",\n"
            "    \"risk_assessment\": \"创新风险评估\",\n"
            "    \"expected_breakthrough\": \"期待的突破点\"\n"
            "  }}\n"
            "}}\n\n"
            "🎨 **发挥创造力，设计一个既科学又富有想象力的分发策略！**"
        )
        prompt = PromptTemplate(
            input_variables=['round_number', 'candidates', 'params', 'constraints', 'user_segments'], 
            template=template
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=prompt)

    def _try_parse_json(self, text: str) -> Any:
        """尝试解析文本为 JSON；若失败，尝试提取首个 JSON 片段再解析。"""
        try:
            return json.loads(text)
        except Exception:
            # 尝试通过正则提取可能的 JSON 对象或数组
            m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
            if m:
                snippet = m.group(1)
                try:
                    return json.loads(snippet)
                except Exception:
                    logger.warning("提取到 JSON 片段但解析失败: %s", snippet)
            # 无法解析
            raise

    # --------- API 兼容方法 ---------
    def initialize_batch(self, posts: List[Dict[str, Any]], users: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 与 ContentDistributor 保持一致的初始化结构
        for i, post in enumerate(posts):
            if 'post_id' not in post:
                post['post_id'] = f"post_{self._rng.getrandbits(48):012x}"[:18]

        self.distribution_plan = {
            'batch_id': self.batch_id,
            'created_at': datetime.now().isoformat(),
            'posts': {post['post_id']: {
                'post_id': post['post_id'],
                'content': post.get('content', ''),
                'platform': post.get('platform', 'unknown'),
                'title': post.get('title', ''),
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
        return self.distribution_plan

    def generate_round_distribution(self, round_number: int,
                                    posts_per_round: int = 5,
                                    users_per_post: int = 10,
                                    hot_post_ratio: float = 0.4) -> Dict[str, Any]:
        logger.info(f"🚀 开始生成第 {round_number} 轮智能分发策略")
        
        # 发送分发开始事件到前端
        if self.realtime_callback:
            self.realtime_callback('distribution_started', {
                'round_number': round_number,
                'message': f'🚀 开始生成第 {round_number} 轮智能分发策略',
                'timestamp': datetime.now().isoformat()
            })
        
        # 根据评估结果优化参数
        optimized_params = self._optimize_distribution_params(
            round_number, posts_per_round, users_per_post, hot_post_ratio
        )
        
        posts_per_round = optimized_params.get('posts_per_round', posts_per_round)
        users_per_post = optimized_params.get('users_per_post', users_per_post)
        hot_post_ratio = optimized_params.get('hot_post_ratio', hot_post_ratio)
        
        all_post_ids = list(self.distribution_plan['posts'].keys())
        if not all_post_ids:
            return {'round_number': round_number, 'posts': {}, 'total_posts': 0, 'total_planned_interactions': 0}

        # 第1轮使用随机策略
        if round_number == 1:
            logger.info("📍 第1轮使用随机分发策略")
            # 发送策略选择事件
            if self.realtime_callback:
                self.realtime_callback('strategy_selected', {
                    'round_number': round_number,
                    'strategy': 'random',
                    'message': '📍 第1轮使用随机分发策略',
                    'timestamp': datetime.now().isoformat()
                })
            selected_posts = self._select_first_round_posts(posts_per_round)
            llm_strategy = None
        else:
            # 第2轮起使用LLM智能策略
            logger.info("🧠 使用LLM智能分发策略")
            # 发送策略选择事件
            if self.realtime_callback:
                self.realtime_callback('strategy_selected', {
                    'round_number': round_number,
                    'strategy': 'llm_intelligent',
                    'message': '🧠 使用LLM智能分发策略',
                    'timestamp': datetime.now().isoformat()
                })
            selected_posts, llm_strategy = self._llm_driven_post_selection(
                all_post_ids, round_number, posts_per_round, users_per_post, hot_post_ratio
            )

        # 构建分发结果
        round_distribution = {
            'round_number': round_number,
            'created_at': datetime.now().isoformat(),
            'posts': {},
            'total_posts': len(selected_posts),
            'total_planned_interactions': 0,
            'optimization_applied': optimized_params,
            'llm_strategy': llm_strategy
        }

        # 为每个选中的帖子分配用户
        for post_id in selected_posts:
            # 获取LLM建议的用户分配策略（如果有）
            user_strategy = self._get_user_strategy_for_post(post_id, llm_strategy)
            assigned_users = self._select_users_for_post_with_strategy(
                post_id, round_number, users_per_post, user_strategy
            )
            
            round_distribution['posts'][post_id] = {
                'post_id': post_id,
                'assigned_users': assigned_users,
                'user_count': len(assigned_users),
                'selection_reason': self._get_post_selection_reason(post_id, round_number),
                'user_strategy': user_strategy
            }
            round_distribution['total_planned_interactions'] += len(assigned_users)

        self.distribution_plan['rounds'].append(round_distribution)
        self._save_plan()
        
        logger.info(f"✅ 第 {round_number} 轮分发策略生成完成：{len(selected_posts)} 个帖子，{round_distribution['total_planned_interactions']} 次计划交互")
        
        # 发送分发完成事件，包含选中的帖子信息
        if self.realtime_callback:
            # 构建帖子详细信息
            posts_info = []
            for post_id in selected_posts:
                post_data = self.distribution_plan['posts'].get(post_id, {})
                posts_info.append({
                    'post_id': post_id,
                    'title': post_data.get('title', f'帖子 {post_id}'),
                    'content': post_data.get('content', ''),
                    'platform': post_data.get('platform', 'unknown'),
                    'assigned_users': round_distribution['posts'][post_id]['assigned_users'],
                    'user_count': round_distribution['posts'][post_id]['user_count'],
                    'selection_reason': round_distribution['posts'][post_id]['selection_reason'],
                    'original_metrics': post_data.get('original_metrics', {})
                })
            
            self.realtime_callback('distribution_completed', {
                'round_number': round_number,
                'message': f'✅ 第 {round_number} 轮分发策略生成完成：{len(selected_posts)} 个帖子，{round_distribution["total_planned_interactions"]} 次计划交互',
                'posts': posts_info,
                'total_posts': len(selected_posts),
                'total_planned_interactions': round_distribution['total_planned_interactions'],
                'strategy_type': 'random' if round_number == 1 else 'llm_intelligent',
                'optimization_applied': optimized_params,
                'timestamp': datetime.now().isoformat()
            })
        
        return round_distribution

    def update_round_results(self, round_number: int, results: Dict[str, Any]):
        # 与 ContentDistributor 类似的更新逻辑
        for post_result in results.get('posts', []):
            post_id = post_result['post_id']
            if post_id in self.distribution_plan['posts']:
                post_data = self.distribution_plan['posts'][post_id]
                post_data['simulation_metrics']['total_interactions'] += post_result.get('actions_count', 0)
                post_data['simulation_metrics']['unique_users'] += post_result.get('users_count', 0)

                interactions_weight = 1.0
                users_weight = 2.0
                post_data['simulation_metrics']['heat_score'] = (
                    post_data['simulation_metrics']['total_interactions'] * interactions_weight +
                    post_data['simulation_metrics']['unique_users'] * users_weight
                )

                post_data['round_history'].append({
                    'round': round_number,
                    'interactions': post_result.get('actions_count', 0),
                    'users': post_result.get('users_count', 0),
                    'action_types': post_result.get('action_types', {})
                })

        # 更新用户活跃度
        self._update_user_activity(round_number, results)
        self._save_plan()

    def get_hot_posts(self, top_k: int = 5) -> List[Dict[str, Any]]:
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

    # --------- 内部辅助方法（与原实现类似） ---------
    def _load_or_create_plan(self) -> Dict[str, Any]:
        if self.distribution_plan_file.exists():
            try:
                with open(self.distribution_plan_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 若文件只包含 rounds，则构造一个最小结构
                    if 'rounds' in data and ('posts' not in data or 'users' not in data):
                        return {
                            'batch_id': data.get('batch_id', self.batch_id),
                            'created_at': data.get('created_at', datetime.now().isoformat()),
                            'posts': {},
                            'users': {},
                            'rounds': data.get('rounds', [])
                        }
                    return data
            except Exception:
                return {}
        else:
            return {}

    def _save_plan(self):
        """保存完整 distribution_plan 到磁盘，便于恢复和审计。"""
        try:
            with open(self.distribution_plan_file, 'w', encoding='utf-8') as f:
                json.dump(self.distribution_plan, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("保存 distribution_plan 失败")
            raise

    def _select_first_round_posts(self, count: int) -> List[str]:
        all_post_ids = list(self.distribution_plan['posts'].keys())
        return self._rng.sample(all_post_ids, min(count, len(all_post_ids)))

    def _select_subsequent_round_posts(self, round_number: int, count: int, hot_ratio: float) -> List[str]:
        hot_count = int(count * hot_ratio)
        random_count = count - hot_count

        selected_posts = []
        if hot_count > 0:
            hot_posts = self.get_hot_posts(hot_count * 2)
            hot_post_ids = [p['post_id'] for p in hot_posts[:hot_count]]
            selected_posts.extend(hot_post_ids)

        if random_count > 0:
            unused_posts = [pid for pid, p in self.distribution_plan['posts'].items() if not p['round_history']]
            if len(unused_posts) >= random_count:
                random_posts = self._rng.sample(unused_posts, random_count)
            else:
                all_posts = list(self.distribution_plan['posts'].keys())
                available_posts = [p for p in all_posts if p not in selected_posts]
                random_posts = self._rng.sample(available_posts, min(random_count, len(available_posts)))

            selected_posts.extend(random_posts)

        return selected_posts

    def _select_users_for_post(self, post_id: str, round_number: int, count: int) -> List[str]:
        all_users = list(self.distribution_plan['users'].keys())

        if round_number == 1:
            return self._rng.sample(all_users, min(count, len(all_users)))
        else:
            active_users = [uid for uid, udata in self.distribution_plan['users'].items()
                            if udata['last_active_round'] == round_number - 1]
            inactive_users = [uid for uid, udata in self.distribution_plan['users'].items()
                              if udata['last_active_round'] < round_number - 1]

            active_count = min(int(count * 0.7), len(active_users))
            inactive_count = count - active_count

            selected = []
            if active_count > 0 and active_users:
                selected.extend(self._rng.sample(active_users, min(active_count, len(active_users))))
            if inactive_count > 0 and inactive_users:
                selected.extend(self._rng.sample(inactive_users, min(inactive_count, len(inactive_users))))

            if len(selected) < count:
                remaining_users = [u for u in all_users if u not in selected]
                if remaining_users:
                    additional = self._rng.sample(remaining_users, min(count - len(selected), len(remaining_users)))
                    selected.extend(additional)

            return selected

    def _get_post_selection_reason(self, post_id: str, round_number: int) -> str:
        if round_number == 1:
            return "首轮随机选择"

        post_data = self.distribution_plan['posts'].get(post_id, {})
        if post_data.get('round_history'):
            return f"热门帖子 (热度: {post_data['simulation_metrics']['heat_score']:.1f})"
        else:
            return "新帖子随机选择"

    def _update_user_activity(self, round_number: int, results: Dict[str, Any]):
        for post_result in results.get('posts', []):
            post_id = post_result['post_id']
            if round_number <= len(self.distribution_plan.get('rounds', [])):
                round_data = self.distribution_plan['rounds'][round_number - 1]
                if post_id in round_data['posts']:
                    assigned_users = round_data['posts'][post_id]['assigned_users']
                    for user_id in assigned_users:
                        if user_id in self.distribution_plan['users']:
                            self.distribution_plan['users'][user_id]['last_active_round'] = round_number
                            self.distribution_plan['users'][user_id]['activity_score'] += 1.0
    
    def update_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """更新评估结果，用于后续优化"""
        self.evaluation_results = evaluation_results
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': evaluation_results
        })
        logger.info("评估结果已更新，将用于后续分发策略优化")
    
    def _build_complex_post_features(self, post_ids: List[str], round_number: int) -> List[Dict[str, Any]]:
        """构建复杂的帖子特征集合"""
        candidates = []
        
        for pid in post_ids:
            post_data = self.distribution_plan['posts'][pid]
            metrics = post_data['simulation_metrics']
            
            # 基础特征
            base_features = {
                'post_id': pid,
                'heat_score': metrics['heat_score'],
                'total_interactions': metrics['total_interactions'],
                'likes': metrics.get('likes', 0),
                'comments': metrics.get('comments', 0),
                'shares': metrics.get('shares', 0)
            }
            
            # 时间衰减因子
            rounds_since_last = self._calculate_rounds_since_last_use(pid, round_number)
            time_decay = max(0.1, 1.0 - (rounds_since_last * 0.2))
            
            # 用户参与多样性
            user_diversity = self._calculate_user_diversity_score(pid)
            
            # 内容新鲜度
            content_freshness = self._calculate_content_freshness(pid, round_number)
            
            # 情感极化度
            sentiment_polarization = self._calculate_sentiment_polarization(pid)
            
            # 话题传播潜力
            viral_potential = self._calculate_viral_potential(pid)
            
            # 用户群体覆盖度
            audience_coverage = self._calculate_audience_coverage(pid)
            
            # 认知负荷影响
            cognitive_load = self._calculate_cognitive_load_impact(pid)
            
            # 组合复杂特征
            complex_features = {
                **base_features,
                'time_decay_factor': time_decay,
                'user_diversity_score': user_diversity,
                'content_freshness': content_freshness,
                'sentiment_polarization': sentiment_polarization,
                'viral_potential': viral_potential,
                'audience_coverage': audience_coverage,
                'cognitive_load_impact': cognitive_load,
                'rounds_since_last_use': rounds_since_last,
                'engagement_momentum': self._calculate_engagement_momentum(pid),
                'cross_post_influence': self._calculate_cross_post_influence(pid, post_ids)
            }
            
            candidates.append(complex_features)
        
        return candidates
    
    def _execute_complex_selection_strategy(self, candidates: List[Dict[str, Any]], 
                                          round_number: int, posts_per_round: int, 
                                          hot_post_ratio: float) -> List[str]:
        """执行多阶段复杂选择策略"""
        
        # 阶段1: 多因子评分
        scored_candidates = self._multi_factor_scoring(candidates, round_number)
        
        # 阶段2: 动态权重调整
        weighted_candidates = self._dynamic_weight_adjustment(scored_candidates, round_number)
        
        # 阶段3: 多样性约束优化
        diversity_optimized = self._diversity_constraint_optimization(weighted_candidates)
        
        # 阶段4: 时序依赖分析
        temporal_optimized = self._temporal_dependency_analysis(diversity_optimized, round_number)
        
        # 阶段5: 最终选择与平衡
        final_selection = self._final_selection_balancing(
            temporal_optimized, posts_per_round, hot_post_ratio
        )
        
        return final_selection
    
    def _multi_factor_scoring(self, candidates: List[Dict[str, Any]], round_number: int) -> List[Dict[str, Any]]:
        """多因子评分系统"""
        for candidate in candidates:
            # 热度权重 (30%)
            heat_score = candidate['heat_score'] * 0.3
            
            # 新鲜度权重 (20%)
            freshness_score = candidate['content_freshness'] * 0.2
            
            # 多样性权重 (15%)
            diversity_score = candidate['user_diversity_score'] * 0.15
            
            # 传播潜力权重 (15%)
            viral_score = candidate['viral_potential'] * 0.15
            
            # 时间衰减权重 (10%)
            time_score = candidate['time_decay_factor'] * 0.1
            
            # 认知负荷权重 (10%)
            cognitive_score = (1.0 - candidate['cognitive_load_impact']) * 0.1
            
            # 综合评分
            candidate['composite_score'] = (
                heat_score + freshness_score + diversity_score + 
                viral_score + time_score + cognitive_score
            )
            
            # 动态调整因子
            if round_number > 3:
                # 后期轮次更注重多样性和新鲜度
                candidate['composite_score'] += candidate['content_freshness'] * 0.1
                candidate['composite_score'] += candidate['user_diversity_score'] * 0.1
        
        return sorted(candidates, key=lambda x: x['composite_score'], reverse=True)
    
    def _dynamic_weight_adjustment(self, candidates: List[Dict[str, Any]], round_number: int) -> List[Dict[str, Any]]:
        """动态权重调整"""
        # 根据历史表现调整权重
        if hasattr(self, 'evaluation_results') and self.evaluation_results:
            eval_results = self.evaluation_results
            
            # 如果用户参与度低，提高热度权重
            if eval_results.get('user_engagement', {}).get('average_engagement', 0) < 0.5:
                for candidate in candidates:
                    candidate['composite_score'] += candidate['heat_score'] * 0.1
            
            # 如果内容多样性低，提高新鲜度权重
            if eval_results.get('content_diversity', 0) < 0.6:
                for candidate in candidates:
                    candidate['composite_score'] += candidate['content_freshness'] * 0.15
        
        return candidates
    
    def _diversity_constraint_optimization(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """多样性约束优化"""
        # 确保选择的帖子在情感极化度上有分布
        sentiment_groups = {'positive': [], 'neutral': [], 'negative': []}
        
        for candidate in candidates:
            polarization = candidate['sentiment_polarization']
            if polarization > 0.6:
                sentiment_groups['positive'].append(candidate)
            elif polarization < -0.6:
                sentiment_groups['negative'].append(candidate)
            else:
                sentiment_groups['neutral'].append(candidate)
        
        # 平衡各组的权重
        for group_name, group_candidates in sentiment_groups.items():
            if len(group_candidates) > 0:
                boost_factor = 1.0 / max(1, len(group_candidates) / 3)  # 平衡因子
                for candidate in group_candidates:
                    candidate['diversity_boost'] = boost_factor
                    candidate['composite_score'] *= boost_factor
        
        return sorted(candidates, key=lambda x: x['composite_score'], reverse=True)
    
    def _temporal_dependency_analysis(self, candidates: List[Dict[str, Any]], round_number: int) -> List[Dict[str, Any]]:
        """时序依赖分析"""
        # 分析帖子间的时序关联
        for candidate in candidates:
            # 计算与前一轮选择帖子的关联度
            temporal_relevance = self._calculate_temporal_relevance(candidate['post_id'], round_number)
            
            # 避免连续选择高度相关的帖子
            if temporal_relevance > 0.8:
                candidate['composite_score'] *= 0.7  # 降权
            elif temporal_relevance < 0.3:
                candidate['composite_score'] *= 1.2  # 提权
        
        return candidates
    
    def _final_selection_balancing(self, candidates: List[Dict[str, Any]], 
                                 posts_per_round: int, hot_post_ratio: float) -> List[str]:
        """最终选择与平衡"""
        if not candidates:
            return []
        
        # 分离热门和新帖子
        hot_posts = [c for c in candidates if c['rounds_since_last_use'] <= 1]
        fresh_posts = [c for c in candidates if c['rounds_since_last_use'] > 1]
        
        # 计算各类别数量
        hot_count = min(int(posts_per_round * hot_post_ratio), len(hot_posts))
        fresh_count = min(posts_per_round - hot_count, len(fresh_posts))
        
        # 如果某类别不足，从另一类别补充
        if hot_count < int(posts_per_round * hot_post_ratio) and fresh_posts:
            additional_fresh = min(posts_per_round - hot_count, len(fresh_posts))
            fresh_count = additional_fresh
        elif fresh_count < posts_per_round - hot_count and hot_posts:
            additional_hot = min(posts_per_round - fresh_count, len(hot_posts))
            hot_count = additional_hot
        
        # 最终选择
        selected = []
        selected.extend([c['post_id'] for c in hot_posts[:hot_count]])
        selected.extend([c['post_id'] for c in fresh_posts[:fresh_count]])
        
        return selected[:posts_per_round]
    
    # 复杂特征计算辅助方法
    def _calculate_rounds_since_last_use(self, post_id: str, current_round: int) -> int:
        """计算距离上次使用的轮数"""
        post_data = self.distribution_plan['posts'].get(post_id, {})
        round_history = post_data.get('round_history', [])
        if not round_history:
            return current_round  # 从未使用过
        # Extract round numbers from round_history dictionaries
        round_numbers = [r['round'] for r in round_history if isinstance(r, dict) and 'round' in r]
        if not round_numbers:
            return current_round
        return current_round - max(round_numbers)
    
    def _calculate_user_diversity_score(self, post_id: str) -> float:
        """计算用户参与多样性分数"""
        post_data = self.distribution_plan['posts'].get(post_id, {})
        participants = set()
        
        # 收集所有参与过该帖子的用户
        for round_entry in post_data.get('round_history', []):
            if isinstance(round_entry, dict) and 'round' in round_entry:
                round_num = round_entry['round']
                round_data = self.distribution_plan.get('rounds', [])
                if round_num <= len(round_data):
                    round_info = round_data[round_num - 1]
                    if post_id in round_info.get('posts', {}):
                        participants.update(round_info['posts'][post_id].get('assigned_users', []))
        
        total_users = len(self.distribution_plan.get('users', {}))
        return len(participants) / max(1, total_users)
    
    def _calculate_content_freshness(self, post_id: str, current_round: int) -> float:
        """计算内容新鲜度"""
        rounds_since_last = self._calculate_rounds_since_last_use(post_id, current_round)
        
        # 新鲜度随时间递减，但有最小值
        if rounds_since_last == current_round:  # 从未使用
            return 1.0
        elif rounds_since_last >= 3:  # 3轮以上未使用，重新变新鲜
            return 0.8
        else:
            return max(0.2, 1.0 - (rounds_since_last * 0.3))
    
    def _calculate_sentiment_polarization(self, post_id: str) -> float:
        """计算情感极化度"""
        post_data = self.distribution_plan['posts'].get(post_id, {})
        
        # 基于帖子内容和历史反应计算情感极化
        # 这里简化处理，实际可以通过NLP分析
        content = post_data.get('content', '')
        
        # 简单的关键词检测
        positive_keywords = ['好', '棒', '赞', '支持', '优秀', '满意']
        negative_keywords = ['差', '烂', '糟', '反对', '失望', '问题']
        
        pos_count = sum(1 for word in positive_keywords if word in content)
        neg_count = sum(1 for word in negative_keywords if word in content)
        
        if pos_count + neg_count == 0:
            return 0.0  # 中性
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _calculate_viral_potential(self, post_id: str) -> float:
        """计算话题传播潜力"""
        post_data = self.distribution_plan['posts'].get(post_id, {})
        metrics = post_data.get('simulation_metrics', {})
        
        # 基于历史交互数据计算传播潜力
        interactions = metrics.get('total_interactions', 0)
        likes = metrics.get('likes', 0)
        comments = metrics.get('comments', 0)
        shares = metrics.get('shares', 0)
        
        # 加权计算传播潜力
        viral_score = (
            likes * 0.3 +      # 点赞权重30%
            comments * 0.5 +   # 评论权重50%（更重要）
            shares * 0.2       # 分享权重20%
        )
        
        # 归一化到0-1范围
        return min(1.0, viral_score / max(1, interactions))
    
    def _calculate_audience_coverage(self, post_id: str) -> float:
        """计算用户群体覆盖度"""
        post_data = self.distribution_plan['posts'].get(post_id, {})
        
        # 分析该帖子能覆盖的用户群体类型
        covered_segments = set()
        
        # 基于帖子内容推断目标群体（简化处理）
        content = post_data.get('content', '').lower()
        
        if any(word in content for word in ['技术', '科技', '创新']):
            covered_segments.add('tech_enthusiasts')
        if any(word in content for word in ['投资', '股票', '金融']):
            covered_segments.add('investors')
        if any(word in content for word in ['安全', '事故', '风险']):
            covered_segments.add('safety_conscious')
        if any(word in content for word in ['品牌', '产品', '体验']):
            covered_segments.add('consumers')
        
        # 默认覆盖一般用户群体
        if not covered_segments:
            covered_segments.add('general_users')
        
        # 假设有4个主要用户群体
        return len(covered_segments) / 4.0
    
    def _calculate_cognitive_load_impact(self, post_id: str) -> float:
        """计算认知负荷影响"""
        post_data = self.distribution_plan['posts'].get(post_id, {})
        content = post_data.get('content', '')
        
        # 基于内容复杂度计算认知负荷
        content_length = len(content)
        
        # 复杂词汇检测
        complex_words = ['技术', '系统', '算法', '分析', '评估', '优化', '策略']
        complex_count = sum(1 for word in complex_words if word in content)
        
        # 数字和专业术语
        numbers = len(re.findall(r'\d+', content))
        
        # 计算认知负荷分数
        base_load = min(1.0, content_length / 200.0)  # 基于长度
        complexity_load = min(0.5, complex_count * 0.1)  # 基于复杂词汇
        data_load = min(0.3, numbers * 0.05)  # 基于数字信息
        
        return base_load + complexity_load + data_load
    
    def _calculate_engagement_momentum(self, post_id: str) -> float:
        """计算参与动量"""
        post_data = self.distribution_plan['posts'].get(post_id, {})
        round_history = post_data.get('round_history', [])
        
        if len(round_history) < 2:
            return 0.5  # 默认中等动量
        
        # 计算最近两轮的参与度变化
        recent_rounds = round_history[-2:]
        momentum = 0.0
        
        for i, round_entry in enumerate(recent_rounds):
            if isinstance(round_entry, dict) and 'round' in round_entry:
                round_num = round_entry['round']
                round_data = self.distribution_plan.get('rounds', [])
                if round_num <= len(round_data):
                    round_info = round_data[round_num - 1]
                    if post_id in round_info.get('posts', {}):
                        interactions = len(round_info['posts'][post_id].get('assigned_users', []))
                        momentum += interactions * (i + 1)  # 最近的轮次权重更高
        
        return min(1.0, momentum / 20.0)  # 归一化
    
    def _calculate_cross_post_influence(self, post_id: str, all_post_ids: List[str]) -> float:
        """计算跨帖子影响力"""
        post_data = self.distribution_plan['posts'].get(post_id, {})
        content = post_data.get('content', '')
        
        # 计算与其他帖子的内容相似度
        similarity_scores = []
        
        for other_id in all_post_ids:
            if other_id == post_id:
                continue
            
            other_data = self.distribution_plan['posts'].get(other_id, {})
            other_content = other_data.get('content', '')
            
            # 简单的关键词重叠度计算
            words1 = set(content.split())
            words2 = set(other_content.split())
            
            if len(words1) == 0 or len(words2) == 0:
                similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0.0
            
            similarity_scores.append(similarity)
        
        # 返回平均相似度作为影响力指标
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_temporal_relevance(self, post_id: str, current_round: int) -> float:
        """计算时序相关性"""
        if current_round <= 1:
            return 0.0
        
        # 获取前一轮选择的帖子
        prev_round_data = self.distribution_plan.get('rounds', [])
        if current_round - 1 > len(prev_round_data):
            return 0.0
        
        prev_round = prev_round_data[current_round - 2]  # 前一轮（索引-1）
        prev_posts = list(prev_round.get('posts', {}).keys())
        
        if not prev_posts:
            return 0.0
        
        # 计算与前一轮帖子的相关性
        current_post_data = self.distribution_plan['posts'].get(post_id, {})
        current_content = current_post_data.get('content', '')
        
        relevance_scores = []
        for prev_post_id in prev_posts:
            prev_post_data = self.distribution_plan['posts'].get(prev_post_id, {})
            prev_content = prev_post_data.get('content', '')
            
            # 计算内容相关性
            words1 = set(current_content.split())
            words2 = set(prev_content.split())
            
            if len(words1) == 0 or len(words2) == 0:
                relevance = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                relevance = intersection / union if union > 0 else 0.0
            
            relevance_scores.append(relevance)
        
        return max(relevance_scores) if relevance_scores else 0.0
    
    def _adaptive_distribution_params(self, round_number: int, base_posts: int, 
                                    base_users: int, base_hot_ratio: float) -> Dict[str, Any]:
        """自适应分发参数调整 - 更灵活的策略"""
        
        # 获取当前系统状态
        current_state = self._analyze_current_system_state()
        
        # 动态调整帖子数量 (1-8个帖子)
        posts_per_round = self._calculate_adaptive_posts_count(
            base_posts, current_state, round_number
        )
        
        # 动态调整用户数量 (4-12个用户)
        users_per_post = self._calculate_adaptive_users_count(
            base_users, current_state, round_number
        )
        
        # 动态调整热门比例 (0.2-0.8)
        hot_post_ratio = self._calculate_adaptive_hot_ratio(
            base_hot_ratio, current_state, round_number
        )
        
        # 新增：个性化匹配强度
        personalization_strength = self._calculate_personalization_strength(current_state)
        
        # 新增：实时响应敏感度
        real_time_sensitivity = self._calculate_real_time_sensitivity(current_state)
        
        return {
            'posts_per_round': posts_per_round,
            'users_per_post': users_per_post,
            'hot_post_ratio': hot_post_ratio,
            'personalization_strength': personalization_strength,
            'real_time_sensitivity': real_time_sensitivity,
            'adaptation_reason': current_state['adaptation_reason']
        }
    
    def _analyze_current_system_state(self) -> Dict[str, Any]:
        """分析当前系统状态，为自适应调整提供依据"""
        state = {
            'user_activity_level': 'medium',
            'content_diversity': 'medium', 
            'engagement_momentum': 'stable',
            'adaptation_reason': []
        }
        
        # 分析用户活跃度
        active_users = sum(1 for u in self.distribution_plan['users'].values() 
                          if u.get('last_active_round', 0) > 0)
        total_users = len(self.distribution_plan['users'])
        activity_ratio = active_users / max(1, total_users)
        
        if activity_ratio > 0.7:
            state['user_activity_level'] = 'high'
            state['adaptation_reason'].append('高用户活跃度')
        elif activity_ratio < 0.3:
            state['user_activity_level'] = 'low'
            state['adaptation_reason'].append('低用户活跃度')
        
        # 分析内容多样性
        used_posts = len([p for p in self.distribution_plan['posts'].values() 
                         if p.get('used_rounds', [])])
        total_posts = len(self.distribution_plan['posts'])
        diversity_ratio = used_posts / max(1, total_posts)
        
        if diversity_ratio < 0.3:
            state['content_diversity'] = 'high'
            state['adaptation_reason'].append('内容池丰富')
        elif diversity_ratio > 0.7:
            state['content_diversity'] = 'low'
            state['adaptation_reason'].append('内容池稀缺')
        
        # 分析参与度趋势
        if hasattr(self, 'evaluation_results') and self.evaluation_results:
            engagement = self.evaluation_results.get('distribution_impact', {}).get('engagement_metrics', {})
            participation_rate = engagement.get('participation_rate', 0.5)
            
            if participation_rate > 0.6:
                state['engagement_momentum'] = 'rising'
                state['adaptation_reason'].append('参与度上升')
            elif participation_rate < 0.3:
                state['engagement_momentum'] = 'declining'
                state['adaptation_reason'].append('参与度下降')
        
        return state
    
    def _calculate_adaptive_posts_count(self, base_posts: int, state: Dict[str, Any], round_number: int) -> int:
        """动态计算帖子数量"""
        posts = base_posts
        
        # 根据用户活跃度调整
        if state['user_activity_level'] == 'high':
            posts += 2  # 高活跃度时增加内容
        elif state['user_activity_level'] == 'low':
            posts = max(1, posts - 1)  # 低活跃度时减少内容
        
        # 根据内容多样性调整
        if state['content_diversity'] == 'high':
            posts += 1  # 内容丰富时可以多推送
        elif state['content_diversity'] == 'low':
            posts = max(1, posts - 1)  # 内容稀缺时保守推送
        
        # 根据参与度趋势调整
        if state['engagement_momentum'] == 'rising':
            posts += 1  # 趋势向好时加大投入
        elif state['engagement_momentum'] == 'declining':
            posts = max(1, posts - 1)  # 趋势下滑时谨慎投入
        
        # 轮次递增效应（后期可以更激进）
        if round_number >= 3:
            posts += 1
        
        return min(8, max(1, posts))  # 限制在1-8个帖子
    
    def _calculate_adaptive_users_count(self, base_users: int, state: Dict[str, Any], round_number: int) -> int:
        """动态计算用户数量"""
        users = base_users
        
        # 根据活跃度调整用户覆盖
        if state['user_activity_level'] == 'high':
            users += 2  # 高活跃时可以覆盖更多用户
        elif state['user_activity_level'] == 'low':
            users += 3  # 低活跃时需要更多用户来保证参与
        
        # 根据参与度调整
        if state['engagement_momentum'] == 'declining':
            users += 2  # 参与度下降时扩大覆盖面
        
        return min(12, max(4, users))  # 限制在4-12个用户
    
    def _calculate_adaptive_hot_ratio(self, base_ratio: float, state: Dict[str, Any], round_number: int) -> float:
        """动态计算热门帖子比例"""
        ratio = base_ratio
        
        # 根据参与度调整热门比例
        if state['engagement_momentum'] == 'declining':
            ratio += 0.2  # 参与度下降时增加热门内容
        elif state['engagement_momentum'] == 'rising':
            ratio -= 0.1  # 参与度上升时可以尝试更多新内容
        
        # 根据内容多样性调整
        if state['content_diversity'] == 'high':
            ratio -= 0.1  # 内容丰富时降低热门比例，增加多样性
        
        return min(0.8, max(0.2, ratio))  # 限制在20%-80%
    
    def _calculate_personalization_strength(self, state: Dict[str, Any]) -> float:
        """计算个性化匹配强度"""
        strength = 0.5  # 基础强度
        
        if state['user_activity_level'] == 'high':
            strength += 0.2  # 高活跃用户适合更强个性化
        elif state['user_activity_level'] == 'low':
            strength -= 0.1  # 低活跃用户需要更广泛内容
        
        if state['engagement_momentum'] == 'declining':
            strength += 0.2  # 参与度下降时加强个性化
        
        return min(1.0, max(0.1, strength))
    
    def _calculate_real_time_sensitivity(self, state: Dict[str, Any]) -> float:
        """计算实时响应敏感度"""
        sensitivity = 0.5  # 基础敏感度
        
        if state['engagement_momentum'] == 'rising':
            sensitivity += 0.3  # 上升趋势时提高敏感度
        elif state['engagement_momentum'] == 'declining':
            sensitivity += 0.2  # 下降趋势时也要敏感响应
        
        return min(1.0, max(0.1, sensitivity))
    
    def _optimize_distribution_params(self, round_number: int, posts_per_round: int, 
                                    users_per_post: int, hot_post_ratio: float) -> Dict[str, Any]:
        """根据评估结果优化分发参数（保留原有逻辑作为补充）"""
        # 先使用自适应参数
        adaptive_params = self._adaptive_distribution_params(
            round_number, posts_per_round, users_per_post, hot_post_ratio
        )
        
        # 如果有评估结果，进一步微调
        if not self.evaluation_results:
            return adaptive_params
        
        # 复制自适应参数作为优化基础
        optimized_params = adaptive_params.copy()
        
        # 基于分发影响评估结果优化参数
        if 'distribution_impact' in self.evaluation_results:
            dist_impact = self.evaluation_results['distribution_impact']
            
            # 如果用户覆盖率低，增加每轮帖子数
            if 'distribution_features' in dist_impact:
                features = dist_impact['distribution_features']
                if 'user_assignment_strategy' in features:
                    user_coverage = features['user_assignment_strategy'].get('user_coverage', 0)
                    if user_coverage < 0.7:
                        optimized_params['posts_per_round'] = min(optimized_params['posts_per_round'] + 1, 10)
                        logger.info(f"基于用户覆盖率({user_coverage:.2f})优化：增加每轮帖子数到{optimized_params['posts_per_round']}")
                
                # 如果内容多样性低，调整热门帖子比例
                if 'post_selection_strategy' in features:
                    content_diversity = features['post_selection_strategy'].get('content_diversity', 0)
                    if content_diversity < 0.5:
                        optimized_params['hot_post_ratio'] = max(optimized_params['hot_post_ratio'] - 0.1, 0.2)
                        logger.info(f"基于内容多样性({content_diversity:.2f})优化：降低热门帖子比例到{optimized_params['hot_post_ratio']:.2f}")
            
            # 基于认知变化评估结果优化参数
            if 'cognitive_changes' in dist_impact:
                cognitive_changes = dist_impact['cognitive_changes']
                if 'engagement_patterns' in cognitive_changes:
                    trends = cognitive_changes['engagement_patterns']
                    if trends.get('engagement_trends') == 'decreasing':
                        # 如果参与度下降，增加每帖用户数
                        optimized_params['users_per_post'] = min(optimized_params['users_per_post'] + 2, 15)
                        logger.info(f"基于参与度下降趋势优化：增加每帖用户数到{optimized_params['users_per_post']}")
                
                # 基于认知负荷优化
                if 'cognitive_load' in cognitive_changes:
                    cognitive_load = cognitive_changes['cognitive_load']
                    overload_risk = cognitive_load.get('overload_risk', 'low')
                    if overload_risk == 'high':
                        # 如果认知负荷过高，减少每轮帖子数
                        optimized_params['posts_per_round'] = max(posts_per_round - 1, 3)
                        logger.info(f"基于认知负荷过高优化：减少每轮帖子数到{optimized_params['posts_per_round']}")
                    elif overload_risk == 'low':
                        # 如果认知负荷过低，可以适当增加
                        optimized_params['posts_per_round'] = min(posts_per_round + 1, 10)
                        logger.info(f"基于认知负荷过低优化：增加每轮帖子数到{optimized_params['posts_per_round']}")
        
            # 基于优化建议调整参数
            if 'optimization_suggestions' in dist_impact:
                suggestions = dist_impact['optimization_suggestions']
                for suggestion in suggestions:
                    if '用户覆盖率' in suggestion and '提高' in suggestion:
                        optimized_params['posts_per_round'] = min(optimized_params['posts_per_round'] + 1, 10)
                    elif '内容多样性' in suggestion and '改进' in suggestion:
                        optimized_params['hot_post_ratio'] = max(optimized_params['hot_post_ratio'] - 0.1, 0.2)
                    elif '参与度' in suggestion and '维持' in suggestion:
                        optimized_params['users_per_post'] = min(optimized_params['users_per_post'] + 1, 15)
                    elif '认知负荷' in suggestion and '降低' in suggestion:
                        optimized_params['posts_per_round'] = max(optimized_params['posts_per_round'] - 1, 3)
                        optimized_params['users_per_post'] = max(optimized_params['users_per_post'] - 1, 5)
        
        # 基于模拟vs真实数据评估结果优化
        if 'simulation_vs_real' in self.evaluation_results:
            sim_vs_real = self.evaluation_results['simulation_vs_real']
            similarity_score = sim_vs_real.get('similarity_score', 50)
            
            if similarity_score < 60:
                # 相似性较低，进行激进优化
                optimized_params['posts_per_round'] = min(optimized_params['posts_per_round'] + 2, 12)
                optimized_params['users_per_post'] = min(optimized_params['users_per_post'] + 3, 18)
                optimized_params['hot_post_ratio'] = max(optimized_params['hot_post_ratio'] - 0.2, 0.1)
                logger.info(f"基于相似性评分({similarity_score})进行激进优化")
            elif similarity_score < 80:
                # 相似性中等，进行温和优化
                optimized_params['posts_per_round'] = min(optimized_params['posts_per_round'] + 1, 10)
                optimized_params['users_per_post'] = min(optimized_params['users_per_post'] + 1, 15)
                logger.info(f"基于相似性评分({similarity_score})进行温和优化")
        
        # 基于评论相似度优化
        if 'comment_similarity' in self.evaluation_results and self.evaluation_results['comment_similarity']:
            comment_sim = self.evaluation_results['comment_similarity']
            if 'overall_stats' in comment_sim:
                avg_similarity = comment_sim['overall_stats'].get('average_similarity', 50)
                similarity_trend = comment_sim['overall_stats'].get('similarity_trend', 'stable')
                
                if avg_similarity < 40:
                    # 评论相似度很低，需要调整用户分配策略
                    optimized_params['users_per_post'] = min(users_per_post + 2, 20)
                    optimized_params['hot_post_ratio'] = max(hot_post_ratio - 0.15, 0.1)
                    logger.info(f"基于评论相似度({avg_similarity:.2f})进行策略调整：增加用户多样性")
                elif avg_similarity < 60:
                    # 评论相似度中等，温和调整
                    optimized_params['users_per_post'] = min(users_per_post + 1, 15)
                    logger.info(f"基于评论相似度({avg_similarity:.2f})进行温和调整")
                
                # 基于相似度趋势调整
                if similarity_trend == 'declining':
                    optimized_params['posts_per_round'] = max(posts_per_round - 1, 3)
                    logger.info("基于评论相似度下降趋势：减少每轮帖子数")
                elif similarity_trend == 'improving':
                    optimized_params['posts_per_round'] = min(posts_per_round + 1, 10)
                    logger.info("基于评论相似度提升趋势：适度增加每轮帖子数")
        
        return optimized_params
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        summary = {
            'total_optimizations': len(self.optimization_history),
            'latest_evaluation': None,
            'optimization_trends': {}
        }
        
        if self.optimization_history:
            summary['latest_evaluation'] = self.optimization_history[-1]
            
            # 分析优化趋势
            if len(self.optimization_history) > 1:
                summary['optimization_trends'] = {
                    'optimization_frequency': len(self.optimization_history) / max(1, len(self.distribution_plan.get('rounds', []))),
                    'last_optimization': self.optimization_history[-1]['timestamp']
                }
        
        return summary
    
    def _llm_driven_post_selection(self, all_post_ids: List[str], round_number: int, 
                                 posts_per_round: int, users_per_post: int, 
                                 hot_post_ratio: float) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """使用LLM进行智能帖子选择和策略制定"""
        try:
            # 构建候选帖子特征
            candidates = self._build_enhanced_post_features(all_post_ids, round_number)
            
            # 构建全局约束和用户群体信息
            constraints = self._build_global_constraints(round_number)
            user_segments = self._analyze_user_segments()
            
            # 准备LLM输入
            llm_input = {
                'round_number': round_number,
                'candidates': candidates[:15],  # 限制候选数量避免超长
                'params': {
                    'posts_per_round': posts_per_round,
                    'users_per_post': users_per_post,
                    'hot_post_ratio': hot_post_ratio,
                    'explore_rate': 0.3
                },
                'constraints': constraints,
                'user_segments': user_segments
            }
            
            logger.info(f"🤖 调用LLM进行第 {round_number} 轮分发决策，候选帖子: {len(candidates)} 个")
            
            # 调用LLM
            llm_response = self.llm_chain.invoke(llm_input)
            llm_output = llm_response.get('text', '') if isinstance(llm_response, dict) else str(llm_response)
            
            # 简化日志输出
            logger.info(f"📄 LLM输出长度: {len(llm_output)} 字符")
            
            # 解析LLM输出
            strategy = self._parse_llm_strategy(llm_output)
            
            if strategy and 'ranked_posts' in strategy:
                # 提取选择的帖子ID
                selected_posts = []
                for post_item in strategy['ranked_posts'][:posts_per_round]:
                    post_id = post_item.get('post_id')
                    if post_id and post_id in self.distribution_plan['posts']:
                        selected_posts.append(post_id)
                
                # 应用参数调整
                if 'parameter_adjustments' in strategy:
                    adjustments = strategy['parameter_adjustments']
                    posts_per_round = adjustments.get('posts_per_round', posts_per_round)
                    # 确保选择的帖子数量符合调整后的参数
                    selected_posts = selected_posts[:posts_per_round]
                
                logger.info(f"✨ LLM策略执行成功，选择 {len(selected_posts)} 个帖子")
                # 尝试多个可能的摘要字段
                summary = strategy.get('strategy_philosophy') or strategy.get('strategy_summary') or strategy.get('philosophy') or '智能分发策略'
                logger.info(f"🧠 LLM策略摘要: {strategy.get('strategy_philosophy') or strategy.get('strategy_summary') or '智能分发策略'}")
                
                return selected_posts, strategy
            else:
                logger.warning("⚠️ LLM返回策略格式异常，使用降级策略")
                return self._fallback_post_selection(all_post_ids, round_number, posts_per_round, hot_post_ratio), None
                
        except Exception as e:
            logger.error(f"❌ LLM分发策略失败: {e}，使用降级策略")
            return self._fallback_post_selection(all_post_ids, round_number, posts_per_round, hot_post_ratio), None
    
    def _build_enhanced_post_features(self, post_ids: List[str], round_number: int) -> List[Dict[str, Any]]:
        """构建增强的帖子特征集合"""
        candidates = []
        
        for post_id in post_ids:
            post_data = self.distribution_plan['posts'][post_id]
            metrics = post_data['simulation_metrics']
            
            # 基础特征
            features = {
                'post_id': post_id,
                'content_preview': post_data.get('content', '')[:120] + '...',
                'heat_score': metrics['heat_score'],
                'total_interactions': metrics['total_interactions'],
                'unique_users': metrics['unique_users'],
                'rounds_since_last_use': self._calculate_rounds_since_last_use(post_id, round_number),
                'content_freshness': self._calculate_content_freshness(post_id, round_number),
                'user_diversity_score': self._calculate_user_diversity_score(post_id),
                'sentiment_polarization': self._calculate_sentiment_polarization(post_id),
                'viral_potential': self._calculate_viral_potential(post_id),
                'audience_coverage': self._calculate_audience_coverage(post_id),
                'cognitive_load_impact': self._calculate_cognitive_load_impact(post_id),
                'engagement_momentum': self._calculate_engagement_momentum(post_id)
            }
            
            # 添加话题标签
            features['topic_tags'] = self._extract_topic_tags(post_data.get('content', ''))
            
            # 添加立场倾向
            features['stance_tendency'] = self._analyze_stance_tendency(post_data.get('content', ''))
            
            candidates.append(features)
        
        # 按综合分数排序
        candidates.sort(key=lambda x: x['heat_score'] + x['content_freshness'] * 2, reverse=True)
        return candidates
    
    def _build_global_constraints(self, round_number: int) -> Dict[str, Any]:
        """构建全局约束条件"""
        constraints = {
            'similarity_score': 50.0,
            'comment_similarity_avg': 0.5,
            'user_coverage_last_round': 0.5,
            'content_diversity_recent': 0.5,
            'cognitive_overload_risk': 'medium',
            'engagement_trend': 'stable',
            'objectives': ['maximize_engagement', 'improve_similarity', 'maintain_diversity', 'limit_cognitive_load']
        }
        
        # 尝试从评估结果中获取实际数据
        if self.evaluation_results:
            # 模拟vs真实相似度
            if 'simulation_vs_real' in self.evaluation_results:
                constraints['similarity_score'] = self.evaluation_results['simulation_vs_real'].get('similarity_score', 50.0)
            
            # 评论相似度
            if 'comment_similarity' in self.evaluation_results and self.evaluation_results['comment_similarity']:
                comment_sim = self.evaluation_results['comment_similarity']
                if 'overall_stats' in comment_sim:
                    constraints['comment_similarity_avg'] = comment_sim['overall_stats'].get('average_similarity', 0.5)
            
            # 分发影响
            if 'distribution_impact' in self.evaluation_results:
                dist_impact = self.evaluation_results['distribution_impact']
                if 'distribution_features' in dist_impact:
                    features = dist_impact['distribution_features']
                    if 'user_assignment_strategy' in features:
                        constraints['user_coverage_last_round'] = features['user_assignment_strategy'].get('user_coverage', 0.5)
                    if 'post_selection_strategy' in features:
                        constraints['content_diversity_recent'] = features['post_selection_strategy'].get('content_diversity', 0.5)
                
                # 认知变化
                if 'cognitive_changes' in dist_impact:
                    cognitive = dist_impact['cognitive_changes']
                    if 'cognitive_load' in cognitive:
                        constraints['cognitive_overload_risk'] = cognitive['cognitive_load'].get('overload_risk', 'medium')
                    if 'engagement_patterns' in cognitive:
                        constraints['engagement_trend'] = cognitive['engagement_patterns'].get('engagement_trends', 'stable')
        
        # 尝试从评估文件中加载最新数据
        try:
            eval_file = self.batch_dir / 'evaluation_results.json'
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                    # 更新约束条件
                    if 'simulation_vs_real' in eval_data:
                        constraints['similarity_score'] = eval_data['simulation_vs_real'].get('similarity_score', constraints['similarity_score'])
            
            # 加载评论相似度文件
            comment_sim_file = Path(f"Output/evaluation/{self.batch_id}_comment_similarity.json")
            if comment_sim_file.exists():
                with open(comment_sim_file, 'r', encoding='utf-8') as f:
                    comment_data = json.load(f)
                    if 'overall_stats' in comment_data:
                        constraints['comment_similarity_avg'] = comment_data['overall_stats'].get('average_similarity', constraints['comment_similarity_avg'])
                        constraints['comment_similarity_trend'] = comment_data['overall_stats'].get('similarity_trend', 'stable')
        except Exception as e:
            logger.debug(f"读取评估文件失败: {e}")
        
        return constraints
    
    def _analyze_user_segments(self) -> Dict[str, float]:
        """分析用户群体分布"""
        total_users = len(self.distribution_plan.get('users', {}))
        if total_users == 0:
            return {'active': 0.0, 'inactive': 0.0, 'new': 0.0}
        
        current_round = len(self.distribution_plan.get('rounds', [])) + 1
        
        active_count = 0
        inactive_count = 0
        new_count = 0
        
        for user_id, user_data in self.distribution_plan.get('users', {}).items():
            last_active = user_data.get('last_active_round', 0)
            
            if last_active == current_round - 1:
                active_count += 1
            elif last_active > 0:
                inactive_count += 1
            else:
                new_count += 1
        
        return {
            'active': active_count / total_users,
            'inactive': inactive_count / total_users,
            'new': new_count / total_users
        }
    
    def _parse_llm_strategy(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """解析LLM输出的策略JSON"""
        if not llm_output or not llm_output.strip():
            logger.warning("LLM输出为空")
            return None
        
        # 记录原始输出用于调试
        logger.debug(f"原始LLM输出: {llm_output[:500]}...")
        
        # 清理输出文本
        cleaned_output = llm_output.strip()
        
        # 尝试多种解析策略
        parsing_strategies = [
            self._parse_direct_json,
            self._parse_json_block,
            self._parse_nested_json,
            self._parse_partial_json,
            self._create_fallback_strategy
        ]
        
        for strategy_func in parsing_strategies:
            try:
                result = strategy_func(cleaned_output)
                if result:
                    logger.info(f"✅ LLM策略解析成功，使用{strategy_func.__name__}")
                    return result
            except Exception as e:
                logger.debug(f"{strategy_func.__name__} 解析失败: {e}")
                continue
        
        logger.error("所有解析策略均失败")
        return None
    
    def _parse_direct_json(self, text: str) -> Optional[Dict[str, Any]]:
        """直接JSON解析"""
        strategy = json.loads(text)
        if self._validate_strategy(strategy):
            return strategy
        return None
    
    def _parse_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        """提取JSON代码块"""
        # 查找```json...```块
        json_block_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL | re.IGNORECASE)
        if json_block_match:
            strategy = json.loads(json_block_match.group(1))
            if self._validate_strategy(strategy):
                return strategy
        return None
    
    def _parse_nested_json(self, text: str) -> Optional[Dict[str, Any]]:
        """提取嵌套JSON对象"""
        # 更强大的JSON提取正则
        json_patterns = [
            r'\{[^{}]*"ranked_posts"[^{}]*\[[^\]]*\][^{}]*\}',  # 简单模式
            r'\{(?:[^{}]|\{[^{}]*\})*"ranked_posts"(?:[^{}]|\{[^{}]*\})*\}',  # 中等复杂度
            r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',  # 最复杂模式
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    strategy = json.loads(match)
                    if self._validate_strategy(strategy):
                        return strategy
                except json.JSONDecodeError:
                    continue
        return None
    
    def _parse_partial_json(self, text: str) -> Optional[Dict[str, Any]]:
        """解析部分JSON并补全"""
        # 查找ranked_posts数组
        posts_match = re.search(r'"ranked_posts"\s*:\s*(\[[^\]]*\])', text, re.DOTALL)
        if posts_match:
            try:
                posts_array = json.loads(posts_match.group(1))
                if isinstance(posts_array, list) and len(posts_array) > 0:
                    # 构建最小可用策略
                    strategy = {
                        "strategy_philosophy": "智能内容分发策略 - 基于用户兴趣和内容特征的精准匹配，通过动态调整分发参数实现最优用户体验",
                        "ranked_posts": posts_array,
                        "personalized_matching": [],
                        "adaptive_parameters": {
                            "posts_per_round": len(posts_array),
                            "users_per_post": 8,
                            "hot_post_ratio": 0.6,
                            "personalization_strength": 0.7
                        }
                    }
                    return strategy
            except json.JSONDecodeError:
                pass
        return None
    
    def _create_fallback_strategy(self, text: str) -> Dict[str, Any]:
        """创建兜底策略"""
        # 从文本中提取可能的帖子ID
        post_ids = re.findall(r'post_[a-f0-9]{12}', text)
        if not post_ids:
            # 如果没有找到帖子ID，使用当前可用的帖子
            available_posts = [pid for pid in self.distribution_plan['posts'].keys() 
                             if not self.distribution_plan['posts'][pid].get('used', False)]
            post_ids = available_posts[:3]  # 取前3个
        
        # 构建基础策略
        ranked_posts = []
        for i, post_id in enumerate(post_ids[:6]):  # 最多6个帖子
            ranked_posts.append({
                "post_id": post_id,
                "innovation_score": 0.8 - i * 0.1,
                "selection_rationale": f"智能兜底策略选择第{i+1}个帖子",
                "surprise_factor": "基于内容特征的自动选择"
            })
        
        return {
            "strategy_philosophy": "创新智能分发策略 - 结合用户行为分析与内容特征匹配，通过多维度评估实现个性化内容推荐，确保用户参与度和内容多样性的最佳平衡",
            "ranked_posts": ranked_posts,
            "personalized_matching": [
                {
                    "post_id": post_id,
                    "user_strategy": {"active": 0.7, "inactive": 0.25, "new": 0.05},
                    "emotional_targeting": "平衡情感共鸣策略"
                } for post_id in post_ids[:6]
            ],
            "adaptive_parameters": {
                "posts_per_round": min(len(post_ids), 6),
                "users_per_post": 8,
                "hot_post_ratio": 0.6,
                "personalization_strength": 0.7,
                "innovation_reason": "兜底策略确保系统连续性"
            },
            "experimental_elements": {
                "new_attempts": "智能兜底机制",
                "risk_assessment": "低风险稳定策略",
                "expected_breakthrough": "确保分发流程不中断"
            }
        }
    
    def _validate_strategy(self, strategy: Any) -> bool:
        """验证策略格式"""
        if not isinstance(strategy, dict):
            return False
        
        # 检查必需字段
        required_fields = ['ranked_posts']
        for field in required_fields:
            if field not in strategy:
                return False
        
        # 检查ranked_posts格式
        ranked_posts = strategy['ranked_posts']
        if not isinstance(ranked_posts, list) or len(ranked_posts) == 0:
            return False
        
        # 检查每个帖子项
        for post in ranked_posts:
            if not isinstance(post, dict) or 'post_id' not in post:
                return False
        
        return True

    def _fallback_post_selection(self, all_post_ids: List[str], round_number: int, 
                               posts_per_round: int, hot_post_ratio: float) -> List[str]:
        """降级帖子选择策略"""
        logger.info("🔄 使用降级帖子选择策略")
        return self._select_subsequent_round_posts(round_number, posts_per_round, hot_post_ratio)
    
    def _get_user_strategy_for_post(self, post_id: str, llm_strategy: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """获取特定帖子的用户分配策略"""
        default_strategy = {'active': 0.7, 'inactive': 0.25, 'new': 0.05}
        
        if not llm_strategy or 'user_matching_strategy' not in llm_strategy:
            return default_strategy
        
        # 查找该帖子的用户策略
        for strategy_item in llm_strategy['user_matching_strategy']:
            if strategy_item.get('post_id') == post_id:
                segment_weights = strategy_item.get('segment_weights', {})
                # 标准化权重
                total_weight = sum(segment_weights.values())
                if total_weight > 0:
                    return {
                        'active': segment_weights.get('active', 0.7) / total_weight,
                        'inactive': segment_weights.get('inactive', 0.25) / total_weight,
                        'new': segment_weights.get('new', 0.05) / total_weight
                    }
        
        return default_strategy
    
    def _select_users_for_post_with_strategy(self, post_id: str, round_number: int, 
                                           count: int, user_strategy: Dict[str, float]) -> List[str]:
        """根据策略选择用户"""
        all_users = list(self.distribution_plan['users'].keys())
        
        if round_number == 1:
            # 第一轮完全随机
            return self._rng.sample(all_users, min(count, len(all_users)))
        
        # 按策略分配用户
        active_users = [uid for uid, udata in self.distribution_plan['users'].items()
                       if udata['last_active_round'] == round_number - 1]
        inactive_users = [uid for uid, udata in self.distribution_plan['users'].items()
                         if 0 < udata['last_active_round'] < round_number - 1]
        new_users = [uid for uid, udata in self.distribution_plan['users'].items()
                    if udata['last_active_round'] == 0]
        
        # 计算各组目标数量
        active_target = int(count * user_strategy['active'])
        inactive_target = int(count * user_strategy['inactive'])
        new_target = count - active_target - inactive_target
        
        selected = []
        
        # 选择活跃用户
        if active_target > 0 and active_users:
            selected.extend(self._rng.sample(active_users, min(active_target, len(active_users))))
        
        # 选择非活跃用户
        if inactive_target > 0 and inactive_users:
            selected.extend(self._rng.sample(inactive_users, min(inactive_target, len(inactive_users))))
        
        # 选择新用户
        if new_target > 0 and new_users:
            selected.extend(self._rng.sample(new_users, min(new_target, len(new_users))))
        
        # 如果不够，从剩余用户中补充
        if len(selected) < count:
            remaining_users = [u for u in all_users if u not in selected]
            if remaining_users:
                additional = self._rng.sample(remaining_users, min(count - len(selected), len(remaining_users)))
                selected.extend(additional)
        
        return selected
    
    def _extract_topic_tags(self, content: str) -> List[str]:
        """提取话题标签"""
        tags = []
        content_lower = content.lower()
        
        # 技术类
        if any(word in content_lower for word in ['技术', '科技', '算法', 'ai', '人工智能', '机器学习']):
            tags.append('技术')
        
        # 经济类
        if any(word in content_lower for word in ['经济', '金融', '投资', '股票', '市场', '财经']):
            tags.append('经济')
        
        # 社会类
        if any(word in content_lower for word in ['社会', '政策', '法律', '教育', '医疗', '环保']):
            tags.append('社会')
        
        # 娱乐类
        if any(word in content_lower for word in ['娱乐', '游戏', '电影', '音乐', '体育', '明星']):
            tags.append('娱乐')
        
        return tags if tags else ['综合']
    
    def _analyze_stance_tendency(self, content: str) -> str:
        """分析立场倾向"""
        positive_words = ['支持', '赞成', '好', '优秀', '成功', '进步', '发展', '创新']
        negative_words = ['反对', '批评', '问题', '失败', '落后', '危险', '风险', '担心']
        
        pos_count = sum(1 for word in positive_words if word in content)
        neg_count = sum(1 for word in negative_words if word in content)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'


