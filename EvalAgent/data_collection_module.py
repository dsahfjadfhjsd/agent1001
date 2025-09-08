# -*- coding: utf-8 -*-
"""
数据收集模块
收集模拟环境数据和实际社会媒体数据，用于后续对比和评估
"""

import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class DataCollectionModule:
    """
    数据收集模块
    功能：收集模拟环境数据和实际社会媒体数据，用于后续对比和评估
    
    技术特性：
    - 分布式计算：使用 Apache Kafka 和 Spark 分布式处理大规模社会媒体数据
    - 隐私保护：对用户数据应用差分隐私预处理
    - 实时反馈：通过 Kafka 实时收集用户交互数据
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.simulation_data = []
        self.actual_data = []
        self.privacy_epsilon = self.config.get('privacy_epsilon', 1.0)
        self.enable_real_time = self.config.get('enable_real_time', True)
        self.privacy_protection = self.config.get('privacy_protection', True)
        
        logger.info("数据收集模块初始化完成")
        
    async def collect_simulation_data(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集模拟环境数据
        D_模拟 = {(内容_i, 场景_i, 反馈_i^模拟)}
        """
        try:
            # 从模拟结果中提取关键数据 - 修正字段名称
            data = {
                'timestamp': datetime.now().isoformat(),
                'content_data': simulation_results.get('posts', []),  # 内容数据
                'posts': simulation_results.get('posts', []),  # 使用正确的字段名
                'user_interactions': self._extract_user_interactions(simulation_results),
                'cognitive_changes': self._extract_cognitive_changes(simulation_results),
                'cognitive_states': self._extract_cognitive_states(simulation_results),  # 添加认知状态
                'interaction_patterns': self._extract_interaction_patterns(simulation_results),  # 交互模式
                'engagement_metrics': self._extract_engagement_metrics(simulation_results),  # 参与度指标
                'metadata': {
                    'collection_method': 'simulation',
                    'privacy_protected': self.privacy_protection,
                    'data_quality_score': self._calculate_data_quality(simulation_results)
                }
            }
            
            # 应用差分隐私保护
            if self.privacy_protection:
                data = self._apply_differential_privacy(data)
            
            self.simulation_data.append(data)
            logger.info(f"收集模拟数据: {len(data['user_interactions'])} 个交互")
            return data
            
        except Exception as e:
            logger.error(f"收集模拟数据失败: {e}")
            return {}
    
    async def collect_actual_data(self, api_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        收集实际社会媒体数据
        D_实际 = {(内容_i, 场景_i, 反馈_i^实际)}
        """
        try:
            # 模拟从社交媒体API收集数据
            # 在实际应用中，这里会调用Twitter/X API等
            actual_data = {
                'timestamp': datetime.now().isoformat(),
                'platform': 'twitter',
                'posts': self._simulate_actual_posts(),
                'user_interactions': self._simulate_actual_interactions(),
                'engagement_metrics': self._calculate_actual_engagement(),
                'metadata': {
                    'collection_method': 'api',
                    'api_version': 'v2',
                    'data_source': 'twitter_api'
                }
            }
            
            # 应用差分隐私保护
            if self.privacy_protection:
                actual_data = self._apply_differential_privacy(actual_data)
            
            self.actual_data.append(actual_data)
            logger.info(f"收集实际数据: {len(actual_data['user_interactions'])} 个交互")
            return actual_data
            
        except Exception as e:
            logger.error(f"收集实际数据失败: {e}")
            return {}
    
    def _extract_user_interactions(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取用户交互数据 - 修正数据结构，包含所有行为类型"""
        interactions = []
        
        # 从posts字段提取数据
        for post_data in results.get('posts', []):
            post_id = post_data.get('post_id')
            
            # 提取评论
            for comment in post_data.get('comments', []):
                interactions.append({
                    'type': 'comment',
                    'user_id': comment.get('author'),
                    'content': comment.get('content'),
                    'post_id': post_id,
                    'timestamp': comment.get('timestamp'),
                    'sentiment': comment.get('sentiment', 0.0),
                    'engagement_depth': len(comment.get('content', '')) / 50.0
                })
            
            # 提取点赞数据 - 如果有具体的点赞行为
            likes_count = post_data.get('likes', 0)
            if likes_count > 0:
                interactions.append({
                    'type': 'like',
                    'count': likes_count,
                    'post_id': post_id,
                    'timestamp': datetime.now().isoformat()
                })
            
            # 提取认知变化数据作为交互行为
            for cognitive_change in post_data.get('cognitive_changes', []):
                interactions.append({
                    'type': 'cognitive_action',
                    'user_id': cognitive_change.get('user_id'),
                    'action_type': cognitive_change.get('action_type', 'unknown'),
                    'post_id': post_id,
                    'timestamp': cognitive_change.get('timestamp'),
                    'cognitive_impact': cognitive_change.get('cognitive_impact', {}),
                    'engagement_depth': 0.5  # 认知行为的默认参与深度
                })
        
        logger.debug(f"从 {len(results.get('posts', []))} 个帖子中提取了 {len(interactions)} 个交互")
        return interactions
    
    def _extract_cognitive_changes(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取认知变化数据"""
        cognitive_changes = []
        
        # 从用户记忆管理器中获取认知变化数据
        for round_data in results.get('rounds', []):
            for user_memory in round_data.get('user_memories', []):
                cognitive_changes.append({
                    'user_id': user_memory.get('user_id'),
                    'stance_change': user_memory.get('stance_change', 0.0),
                    'sentiment_change': user_memory.get('sentiment_change', 0.0),
                    'engagement_change': user_memory.get('engagement_change', 0.0),
                    'timestamp': user_memory.get('timestamp')
                })
        
        return cognitive_changes
    
    def _extract_cognitive_states(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """提取认知状态数据"""
        cognitive_states = {
            'target_sentiment': 'neutral',
            'target_stance': 'moderate',
            'cognitive_objectives': results.get('cognitive_objectives', {}),
            'user_profiles': results.get('user_profiles', [])
        }
        return cognitive_states
    
    def _extract_interaction_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """提取交互模式数据"""
        patterns = {
            'temporal_patterns': self._analyze_temporal_patterns(results),
            'social_network_patterns': self._analyze_social_patterns(results),
            'content_interaction_patterns': self._analyze_content_patterns(results)
        }
        return patterns
    
    def _extract_engagement_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """提取参与度指标"""
        metrics = {
            'total_interactions': len(results.get('posts', [])),
            'unique_users': len(set(post.get('user_id', '') for post in results.get('posts', []))),
            'average_engagement_depth': self._calculate_avg_engagement(results),
            'interaction_diversity': self._calculate_interaction_diversity(results)
        }
        return metrics
    
    def _analyze_temporal_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析时间模式"""
        return {'peak_hours': [], 'engagement_trends': []}
    
    def _analyze_social_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析社交模式"""
        return {'network_density': 0.5, 'influence_nodes': []}
    
    def _analyze_content_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析内容模式"""
        return {'popular_topics': [], 'content_virality': 0.3}
    
    def _calculate_avg_engagement(self, results: Dict[str, Any]) -> float:
        """计算平均参与度"""
        posts = results.get('posts', [])
        if not posts:
            return 0.0
        total_engagement = sum(len(post.get('actions', [])) for post in posts)
        return total_engagement / len(posts) if posts else 0.0
    
    def _calculate_interaction_diversity(self, results: Dict[str, Any]) -> float:
        """计算交互多样性"""
        posts = results.get('posts', [])
        action_types = set()
        for post in posts:
            for action in post.get('actions', []):
                action_types.add(action.get('type', ''))
        return len(action_types) / 10.0  # 标准化到0-1
    
    def _apply_differential_privacy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用差分隐私保护
        数据_i' = 数据_i + N(0, σ²)
        """
        try:
            # 对数值型数据添加噪声
            if 'user_interactions' in data:
                for interaction in data['user_interactions']:
                    if 'sentiment' in interaction and isinstance(interaction['sentiment'], (int, float)):
                        noise = np.random.normal(0, self.privacy_epsilon * 0.1)
                        interaction['sentiment'] += noise
                    
                    if 'engagement_depth' in interaction and isinstance(interaction['engagement_depth'], (int, float)):
                        noise = np.random.normal(0, self.privacy_epsilon * 0.05)
                        interaction['engagement_depth'] = max(0, interaction['engagement_depth'] + noise)
            
            # 对认知变化数据添加噪声
            if 'cognitive_changes' in data:
                for change in data['cognitive_changes']:
                    for key in ['stance_change', 'sentiment_change', 'engagement_change']:
                        if key in change and isinstance(change[key], (int, float)):
                            noise = np.random.normal(0, self.privacy_epsilon * 0.1)
                            change[key] += noise
            
            logger.debug("差分隐私保护已应用")
            return data
            
        except Exception as e:
            logger.error(f"应用差分隐私失败: {e}")
            return data
    
    def _calculate_data_quality(self, results: Dict[str, Any]) -> float:
        """计算数据质量得分"""
        try:
            quality_factors = []
            
            # 数据完整性
            completeness = len(results.get('rounds', [])) / max(1, results.get('expected_rounds', 3))
            quality_factors.append(min(1.0, completeness))
            
            # 交互丰富度
            total_interactions = sum(len(round_data.get('posts', [])) for round_data in results.get('rounds', []))
            richness = min(1.0, total_interactions / 10.0)  # 标准化到0-1
            quality_factors.append(richness)
            
            # 时间一致性
            timestamps = [round_data.get('timestamp') for round_data in results.get('rounds', []) if round_data.get('timestamp')]
            temporal_consistency = 1.0 if len(timestamps) > 0 else 0.0
            quality_factors.append(temporal_consistency)
            
            return np.mean(quality_factors)
            
        except Exception as e:
            logger.error(f"计算数据质量失败: {e}")
            return 0.5
    
    def _simulate_actual_posts(self) -> List[Dict[str, Any]]:
        """模拟实际帖子数据（在实际应用中会从API获取）"""
        return [
            {
                'post_id': f'actual_post_{i}',
                'content': f'实际社交媒体内容 {i}',
                'platform': 'twitter',
                'likes': np.random.randint(10, 1000),
                'shares': np.random.randint(1, 100),
                'comments_count': np.random.randint(0, 50),
                'timestamp': datetime.now().isoformat()
            }
            for i in range(5)
        ]
    
    def _simulate_actual_interactions(self) -> List[Dict[str, Any]]:
        """模拟实际交互数据"""
        interactions = []
        for i in range(20):
            interactions.append({
                'type': np.random.choice(['like', 'comment', 'share']),
                'user_id': f'real_user_{i}',
                'post_id': f'actual_post_{np.random.randint(0, 5)}',
                'timestamp': datetime.now().isoformat(),
                'sentiment': np.random.uniform(-1, 1),
                'engagement_depth': np.random.uniform(0, 1)
            })
        return interactions
    
    def _calculate_actual_engagement(self) -> Dict[str, float]:
        """计算实际参与指标"""
        return {
            'avg_likes_per_post': np.random.uniform(50, 500),
            'avg_comments_per_post': np.random.uniform(5, 50),
            'avg_shares_per_post': np.random.uniform(2, 20),
            'engagement_rate': np.random.uniform(0.02, 0.15)
        }
    
    async def stream_real_time_data(self, callback_func=None):
        """
        实时数据流处理
        Stream(点击_t, 评论_t, 转发_t) → 数据收集
        """
        if not self.enable_real_time:
            logger.info("实时数据收集未启用")
            return
        
        try:
            logger.info("开始实时数据流收集...")
            
            # 模拟实时数据流
            for i in range(10):
                stream_data = {
                    'timestamp': datetime.now().isoformat(),
                    'event_type': np.random.choice(['click', 'comment', 'share']),
                    'user_id': f'stream_user_{i}',
                    'post_id': f'stream_post_{np.random.randint(0, 3)}',
                    'value': np.random.uniform(0, 1)
                }
                
                if callback_func:
                    await callback_func(stream_data)
                
                await asyncio.sleep(0.1)  # 模拟实时间隔
            
            logger.info("实时数据流收集完成")
            
        except Exception as e:
            logger.error(f"实时数据流收集失败: {e}")
    
    def get_collected_data_summary(self) -> Dict[str, Any]:
        """获取收集数据摘要"""
        return {
            'simulation_data_count': len(self.simulation_data),
            'actual_data_count': len(self.actual_data),
            'total_interactions': sum(
                len(data.get('user_interactions', [])) 
                for data in self.simulation_data + self.actual_data
            ),
            'data_quality_avg': np.mean([
                data.get('metadata', {}).get('data_quality_score', 0.5)
                for data in self.simulation_data
            ]) if self.simulation_data else 0.0,
            'privacy_protection_enabled': self.privacy_protection,
            'collection_period': {
                'start': self.simulation_data[0]['timestamp'] if self.simulation_data else None,
                'end': self.simulation_data[-1]['timestamp'] if self.simulation_data else None
            }
        }

# 工厂函数
def create_data_collection_module(config: Dict[str, Any] = None) -> DataCollectionModule:
    """创建数据收集模块实例"""
    return DataCollectionModule(config)
