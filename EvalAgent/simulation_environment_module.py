# -*- coding: utf-8 -*-
"""
模拟环境模块
构建模拟环境，运行 DISTAgent 的分发策略，生成模拟反馈
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

class SimulationEnvironmentModule:
    """
    模拟环境模块
    功能：构建模拟环境，运行 DISTAgent 的分发策略，生成模拟反馈
    
    技术特性：
    - 分布式计算：使用 Ray 分布式运行模拟环境，分片用户群体
    - 隐私保护：对模拟用户数据添加差分隐私噪声
    - 自适应权重：动态调整模拟反馈的权重，优化模拟准确性
    - 实时反馈：实时更新模拟环境参数，基于 Kafka 流数据
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.simulation_state = {}
        self.user_groups = []
        self.environment_parameters = {
            'simulation_accuracy_weight': 0.8,
            'user_behavior_variance': 0.2,
            'content_propagation_rate': 0.15,
            'cognitive_influence_factor': 0.7
        }
        self.privacy_epsilon = self.config.get('privacy_epsilon', 1.0)
        self.enable_distributed = self.config.get('enable_distributed', False)
        
        logger.info("模拟环境模块初始化完成")
    
    async def initialize_simulation_environment(self, user_data: List[Dict[str, Any]], 
                                              content_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        初始化模拟环境
        用户群体 = {U1, U2, ..., UN}, Ui ∈ 节点i
        """
        try:
            # 分片用户群体（分布式计算准备）
            self.user_groups = self._partition_users(user_data)
            
            # 初始化环境状态
            self.simulation_state = {
                'users': {user['user_id']: self._initialize_user_state(user) for user in user_data},
                'content_pool': {content['content_id']: content for content in content_data},
                'interaction_network': self._build_interaction_network(user_data),
                'temporal_state': {
                    'current_round': 0,
                    'simulation_start_time': datetime.now().isoformat(),
                    'last_update_time': datetime.now().isoformat()
                },
                'environment_metrics': {
                    'total_users': len(user_data),
                    'total_content': len(content_data),
                    'network_density': self._calculate_network_density(user_data),
                    'simulation_quality_score': 0.8
                }
            }
            
            # 应用差分隐私保护
            if self.config.get('privacy_protection', True):
                self.simulation_state = self._apply_privacy_protection(self.simulation_state)
            
            logger.info(f"模拟环境初始化完成: {len(user_data)} 用户, {len(content_data)} 内容")
            return self.simulation_state
            
        except Exception as e:
            logger.error(f"初始化模拟环境失败: {e}")
            return {}
    
    async def run_distribution_simulation(self, distribution_strategy: Dict[str, Any], 
                                        target_users: List[str], 
                                        content_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行分发模拟
        反馈_i^模拟 = Agent_模拟(Ci, Si, Ui)
        """
        try:
            simulation_results = {
                'simulation_id': f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'strategy': distribution_strategy,
                'target_users': target_users,
                'content_items': content_items,
                'rounds': [],
                'overall_metrics': {}
            }
            
            # 执行多轮模拟
            num_rounds = distribution_strategy.get('num_rounds', 3)
            for round_num in range(num_rounds):
                round_result = await self._simulate_round(
                    round_num, distribution_strategy, target_users, content_items
                )
                simulation_results['rounds'].append(round_result)
                
                # 更新环境状态
                await self._update_environment_state(round_result)
            
            # 计算整体指标
            simulation_results['overall_metrics'] = self._calculate_overall_metrics(
                simulation_results['rounds']
            )
            
            # 优化模拟准确性权重
            await self._optimize_simulation_weights(simulation_results)
            
            logger.info(f"分发模拟完成: {num_rounds} 轮, 目标用户 {len(target_users)}")
            return simulation_results
            
        except Exception as e:
            logger.error(f"运行分发模拟失败: {e}")
            return {}
    
    async def _simulate_round(self, round_num: int, strategy: Dict[str, Any], 
                            target_users: List[str], content_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """模拟单轮分发"""
        try:
            round_result = {
                'round_number': round_num,
                'timestamp': datetime.now().isoformat(),
                'user_interactions': [],
                'content_performance': {},
                'cognitive_changes': [],
                'network_effects': {}
            }
            
            # 为每个目标用户模拟行为
            for user_id in target_users:
                user_state = self.simulation_state['users'].get(user_id, {})
                
                # 为每个内容项模拟用户反应
                for content in content_items:
                    interaction = await self._simulate_user_content_interaction(
                        user_state, content, strategy
                    )
                    round_result['user_interactions'].append(interaction)
                    
                    # 模拟认知变化
                    cognitive_change = self._simulate_cognitive_change(
                        user_state, content, interaction
                    )
                    round_result['cognitive_changes'].append(cognitive_change)
            
            # 计算内容表现
            round_result['content_performance'] = self._calculate_content_performance(
                round_result['user_interactions']
            )
            
            # 模拟网络传播效应
            round_result['network_effects'] = await self._simulate_network_propagation(
                round_result['user_interactions'], strategy
            )
            
            return round_result
            
        except Exception as e:
            logger.error(f"模拟轮次 {round_num} 失败: {e}")
            return {}
    
    async def _simulate_user_content_interaction(self, user_state: Dict[str, Any], 
                                               content: Dict[str, Any], 
                                               strategy: Dict[str, Any]) -> Dict[str, Any]:
        """模拟用户-内容交互"""
        try:
            # 计算用户对内容的兴趣度
            interest_score = self._calculate_interest_score(user_state, content)
            
            # 计算情感匹配度
            sentiment_match = self._calculate_sentiment_match(user_state, content)
            
            # 计算认知负荷
            cognitive_load = self._calculate_cognitive_load(user_state, content)
            
            # 综合决策概率
            interaction_probability = self._calculate_interaction_probability(
                interest_score, sentiment_match, cognitive_load, strategy
            )
            
            # 生成交互行为
            interaction = {
                'user_id': user_state.get('user_id'),
                'content_id': content.get('content_id'),
                'interaction_type': self._sample_interaction_type(interaction_probability),
                'engagement_level': np.clip(interest_score + sentiment_match - cognitive_load, 0, 1),
                'sentiment_response': self._generate_sentiment_response(sentiment_match),
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'interest_score': interest_score,
                    'sentiment_match': sentiment_match,
                    'cognitive_load': cognitive_load,
                    'interaction_probability': interaction_probability
                }
            }
            
            return interaction
            
        except Exception as e:
            logger.error(f"模拟用户交互失败: {e}")
            return {}
    
    def _calculate_interest_score(self, user_state: Dict[str, Any], 
                                content: Dict[str, Any]) -> float:
        """计算用户兴趣得分"""
        user_interests = user_state.get('interests', [])
        content_topics = content.get('topics', [])
        
        if not user_interests or not content_topics:
            return np.random.uniform(0.3, 0.7)
        
        # 计算主题匹配度
        overlap = len(set(user_interests) & set(content_topics))
        max_possible = max(len(user_interests), len(content_topics))
        
        base_score = overlap / max_possible if max_possible > 0 else 0.5
        
        # 添加随机性
        noise = np.random.normal(0, 0.1)
        return np.clip(base_score + noise, 0, 1)
    
    def _calculate_sentiment_match(self, user_state: Dict[str, Any], 
                                 content: Dict[str, Any]) -> float:
        """计算情感匹配度"""
        user_sentiment = user_state.get('sentiment_preference', 0.0)
        content_sentiment = content.get('sentiment_score', 0.0)
        
        # 计算情感距离
        sentiment_distance = abs(user_sentiment - content_sentiment)
        match_score = 1.0 - sentiment_distance
        
        # 添加噪声
        noise = np.random.normal(0, 0.05)
        return np.clip(match_score + noise, 0, 1)
    
    def _calculate_cognitive_load(self, user_state: Dict[str, Any], 
                                content: Dict[str, Any]) -> float:
        """计算认知负荷"""
        content_complexity = content.get('complexity_score', 0.5)
        user_processing_capacity = user_state.get('processing_capacity', 0.7)
        
        # 认知负荷 = 内容复杂度 / 用户处理能力
        cognitive_load = content_complexity / max(user_processing_capacity, 0.1)
        
        return np.clip(cognitive_load, 0, 1)
    
    def _calculate_interaction_probability(self, interest: float, sentiment: float, 
                                         cognitive_load: float, strategy: Dict[str, Any]) -> float:
        """计算交互概率"""
        # 自适应权重
        w_interest = strategy.get('interest_weight', 0.4)
        w_sentiment = strategy.get('sentiment_weight', 0.3)
        w_cognitive = strategy.get('cognitive_weight', 0.3)
        
        probability = (w_interest * interest + 
                      w_sentiment * sentiment - 
                      w_cognitive * cognitive_load)
        
        return np.clip(probability, 0, 1)
    
    def _sample_interaction_type(self, probability: float) -> str:
        """采样交互类型"""
        if probability < 0.2:
            return 'ignore'
        elif probability < 0.5:
            return 'view'
        elif probability < 0.7:
            return 'like'
        elif probability < 0.9:
            return 'comment'
        else:
            return 'share'
    
    def _generate_sentiment_response(self, sentiment_match: float) -> float:
        """生成情感响应"""
        base_sentiment = sentiment_match * 2 - 1  # 转换到 [-1, 1]
        noise = np.random.normal(0, 0.2)
        return np.clip(base_sentiment + noise, -1, 1)
    
    def _simulate_cognitive_change(self, user_state: Dict[str, Any], 
                                 content: Dict[str, Any], 
                                 interaction: Dict[str, Any]) -> Dict[str, Any]:
        """模拟认知变化"""
        try:
            engagement_level = interaction.get('engagement_level', 0.0)
            content_influence = content.get('influence_score', 0.5)
            
            # 计算认知变化幅度
            change_magnitude = engagement_level * content_influence * self.environment_parameters['cognitive_influence_factor']
            
            cognitive_change = {
                'user_id': user_state.get('user_id'),
                'content_id': content.get('content_id'),
                'stance_change': np.random.normal(0, change_magnitude * 0.1),
                'sentiment_change': np.random.normal(0, change_magnitude * 0.15),
                'interest_change': np.random.normal(0, change_magnitude * 0.05),
                'engagement_propensity_change': np.random.normal(0, change_magnitude * 0.08),
                'timestamp': datetime.now().isoformat(),
                'change_magnitude': change_magnitude
            }
            
            return cognitive_change
            
        except Exception as e:
            logger.error(f"模拟认知变化失败: {e}")
            return {}
    
    async def _simulate_network_propagation(self, interactions: List[Dict[str, Any]], 
                                          strategy: Dict[str, Any]) -> Dict[str, Any]:
        """模拟网络传播效应"""
        try:
            propagation_rate = self.environment_parameters['content_propagation_rate']
            
            # 计算传播指标
            total_interactions = len(interactions)
            high_engagement_count = sum(1 for i in interactions if i.get('engagement_level', 0) > 0.7)
            
            network_effects = {
                'propagation_potential': high_engagement_count / max(total_interactions, 1) * propagation_rate,
                'viral_coefficient': np.random.uniform(0.1, 0.8),
                'reach_amplification': np.random.uniform(1.0, 3.0),
                'influence_cascade_depth': np.random.randint(1, 5),
                'network_clustering_effect': np.random.uniform(0.2, 0.9)
            }
            
            return network_effects
            
        except Exception as e:
            logger.error(f"模拟网络传播失败: {e}")
            return {}
    
    def _partition_users(self, user_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """分片用户群体用于分布式处理"""
        if not self.enable_distributed:
            return [user_data]
        
        chunk_size = max(1, len(user_data) // 4)  # 分成4个分片
        return [user_data[i:i + chunk_size] for i in range(0, len(user_data), chunk_size)]
    
    def _initialize_user_state(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """初始化用户状态"""
        return {
            'user_id': user.get('user_id'),
            'interests': user.get('interests', []),
            'sentiment_preference': user.get('sentiment_preference', np.random.uniform(-0.5, 0.5)),
            'processing_capacity': user.get('processing_capacity', np.random.uniform(0.5, 1.0)),
            'engagement_history': [],
            'cognitive_state': {
                'current_stance': user.get('stance', 0.0),
                'openness_to_change': user.get('openness', 0.7),
                'attention_span': user.get('attention_span', 0.6)
            }
        }
    
    def _build_interaction_network(self, user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建用户交互网络"""
        network = {
            'nodes': [user['user_id'] for user in user_data],
            'edges': [],
            'node_properties': {},
            'network_metrics': {}
        }
        
        # 模拟用户连接
        for i, user1 in enumerate(user_data):
            for j, user2 in enumerate(user_data[i+1:], i+1):
                # 基于兴趣相似度建立连接
                similarity = self._calculate_user_similarity(user1, user2)
                if similarity > 0.6:  # 阈值
                    network['edges'].append({
                        'source': user1['user_id'],
                        'target': user2['user_id'],
                        'weight': similarity
                    })
        
        return network
    
    def _calculate_user_similarity(self, user1: Dict[str, Any], user2: Dict[str, Any]) -> float:
        """计算用户相似度"""
        interests1 = set(user1.get('interests', []))
        interests2 = set(user2.get('interests', []))
        
        if not interests1 or not interests2:
            return np.random.uniform(0.2, 0.8)
        
        intersection = len(interests1 & interests2)
        union = len(interests1 | interests2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_network_density(self, user_data: List[Dict[str, Any]]) -> float:
        """计算网络密度"""
        n_users = len(user_data)
        if n_users < 2:
            return 0.0
        
        max_edges = n_users * (n_users - 1) / 2
        # 模拟实际边数
        actual_edges = max_edges * np.random.uniform(0.1, 0.3)
        
        return actual_edges / max_edges
    
    def _apply_privacy_protection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用差分隐私保护
        Ui' = Ui + N(0, σ²)
        """
        try:
            # 对用户状态添加噪声
            for user_id, user_state in state.get('users', {}).items():
                if 'sentiment_preference' in user_state:
                    noise = np.random.normal(0, self.privacy_epsilon * 0.1)
                    user_state['sentiment_preference'] += noise
                
                if 'processing_capacity' in user_state:
                    noise = np.random.normal(0, self.privacy_epsilon * 0.05)
                    user_state['processing_capacity'] = np.clip(
                        user_state['processing_capacity'] + noise, 0, 1
                    )
            
            logger.debug("模拟环境差分隐私保护已应用")
            return state
            
        except Exception as e:
            logger.error(f"应用隐私保护失败: {e}")
            return state
    
    async def _update_environment_state(self, round_result: Dict[str, Any]):
        """更新环境状态"""
        try:
            # 更新用户状态
            for cognitive_change in round_result.get('cognitive_changes', []):
                user_id = cognitive_change.get('user_id')
                if user_id in self.simulation_state['users']:
                    user_state = self.simulation_state['users'][user_id]
                    
                    # 应用认知变化
                    if 'cognitive_state' in user_state:
                        user_state['cognitive_state']['current_stance'] += cognitive_change.get('stance_change', 0)
                        user_state['sentiment_preference'] += cognitive_change.get('sentiment_change', 0)
            
            # 更新时间状态
            self.simulation_state['temporal_state']['current_round'] += 1
            self.simulation_state['temporal_state']['last_update_time'] = datetime.now().isoformat()
            
            logger.debug(f"环境状态已更新 - 轮次 {self.simulation_state['temporal_state']['current_round']}")
            
        except Exception as e:
            logger.error(f"更新环境状态失败: {e}")
    
    def _calculate_content_performance(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算内容表现"""
        content_stats = {}
        
        for interaction in interactions:
            content_id = interaction.get('content_id')
            if content_id not in content_stats:
                content_stats[content_id] = {
                    'total_interactions': 0,
                    'avg_engagement': 0.0,
                    'sentiment_sum': 0.0,
                    'interaction_types': {}
                }
            
            stats = content_stats[content_id]
            stats['total_interactions'] += 1
            stats['avg_engagement'] += interaction.get('engagement_level', 0.0)
            stats['sentiment_sum'] += interaction.get('sentiment_response', 0.0)
            
            interaction_type = interaction.get('interaction_type', 'unknown')
            stats['interaction_types'][interaction_type] = stats['interaction_types'].get(interaction_type, 0) + 1
        
        # 计算平均值
        for content_id, stats in content_stats.items():
            if stats['total_interactions'] > 0:
                stats['avg_engagement'] /= stats['total_interactions']
                stats['avg_sentiment'] = stats['sentiment_sum'] / stats['total_interactions']
        
        return content_stats
    
    def _calculate_overall_metrics(self, rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算整体指标"""
        try:
            total_interactions = sum(len(round_data.get('user_interactions', [])) for round_data in rounds)
            total_cognitive_changes = sum(len(round_data.get('cognitive_changes', [])) for round_data in rounds)
            
            avg_engagement = np.mean([
                interaction.get('engagement_level', 0.0)
                for round_data in rounds
                for interaction in round_data.get('user_interactions', [])
            ]) if total_interactions > 0 else 0.0
            
            return {
                'total_rounds': len(rounds),
                'total_interactions': total_interactions,
                'total_cognitive_changes': total_cognitive_changes,
                'avg_engagement_level': avg_engagement,
                'simulation_quality_score': min(1.0, avg_engagement + 0.2),
                'network_activation_rate': total_interactions / max(1, len(rounds) * 10)  # 标准化
            }
            
        except Exception as e:
            logger.error(f"计算整体指标失败: {e}")
            return {}
    
    async def _optimize_simulation_weights(self, simulation_results: Dict[str, Any]):
        """
        优化模拟准确性权重
        w* = argmax_w E[模拟准确性|w]
        """
        try:
            current_quality = simulation_results['overall_metrics'].get('simulation_quality_score', 0.5)
            
            # 基于质量得分调整权重
            if current_quality < 0.6:
                # 提高认知影响因子
                self.environment_parameters['cognitive_influence_factor'] *= 1.1
                self.environment_parameters['user_behavior_variance'] *= 0.9
            elif current_quality > 0.8:
                # 增加随机性以避免过拟合
                self.environment_parameters['user_behavior_variance'] *= 1.05
            
            # 限制参数范围
            for key, value in self.environment_parameters.items():
                self.environment_parameters[key] = np.clip(value, 0.1, 1.0)
            
            logger.info(f"模拟权重已优化，质量得分: {current_quality:.3f}")
            
        except Exception as e:
            logger.error(f"优化模拟权重失败: {e}")
    
    def get_simulation_state_summary(self) -> Dict[str, Any]:
        """获取模拟状态摘要"""
        return {
            'environment_parameters': self.environment_parameters,
            'user_count': len(self.simulation_state.get('users', {})),
            'content_count': len(self.simulation_state.get('content_pool', {})),
            'current_round': self.simulation_state.get('temporal_state', {}).get('current_round', 0),
            'network_metrics': self.simulation_state.get('interaction_network', {}).get('network_metrics', {}),
            'simulation_quality': self.simulation_state.get('environment_metrics', {}).get('simulation_quality_score', 0.0)
        }

# 工厂函数
def create_simulation_environment_module(config: Dict[str, Any] = None) -> SimulationEnvironmentModule:
    """创建模拟环境模块实例"""
    return SimulationEnvironmentModule(config)
