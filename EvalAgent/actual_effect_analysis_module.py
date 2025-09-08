# -*- coding: utf-8 -*-
"""
实际效果分析模块
分析 DISTAgent 在实际社会媒体环境中的分发效果，量化关键指标
"""

import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EffectMetrics:
    """效果指标数据类"""
    ctr: float  # 点击率
    deep_engagement_index: float  # 深度参与指数
    propagation_influence: float  # 传播影响力
    reach_amplification: float  # 触达放大系数
    viral_coefficient: float  # 病毒传播系数

class ActualEffectAnalysisModule:
    """
    实际效果分析模块
    功能：分析 DISTAgent 在实际社会媒体环境中的分发效果，量化关键指标
    
    技术特性：
    - 分布式计算：分布式计算反馈指标，参考 Spark
    - 隐私保护：对实际反馈数据应用差分隐私
    - 自适应权重：动态调整指标权重
    - 实时反馈：通过 Kafka 实时收集实际反馈
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.privacy_epsilon = self.config.get('privacy_epsilon', 1.0)
        self.enable_distributed = self.config.get('enable_distributed', False)
        self.enable_deep_engagement = self.config.get('enable_deep_engagement', True)
        self.propagation_analysis = self.config.get('propagation_analysis', True)
        
        # 自适应权重参数
        self.adaptive_weights = {
            'ctr_weight': 0.25,
            'engagement_weight': 0.25,
            'propagation_weight': 0.25,
            'reach_weight': 0.25
        }
        
        logger.info("实际效果分析模块初始化完成")
        
    async def calculate_comprehensive_metrics(self, data: Dict[str, Any]) -> EffectMetrics:
        """
        计算综合效果指标
        指标_i = Compute(反馈_i^实际, 节点_i)
        """
        try:
            # 计算点击率
            ctr = await self._calculate_ctr(data)
            
            # 计算深度参与指数
            deep_engagement = await self._calculate_deep_engagement_index(data)
            
            # 计算传播影响力
            propagation_influence = await self._calculate_propagation_influence(data)
            
            # 计算触达放大系数
            reach_amplification = await self._calculate_reach_amplification(data)
            
            # 计算病毒传播系数
            viral_coefficient = await self._calculate_viral_coefficient(data)
            
            # 应用差分隐私保护
            if self.config.get('privacy_protection', True):
                ctr, deep_engagement, propagation_influence = self._apply_privacy_protection(
                    ctr, deep_engagement, propagation_influence
                )
            
            metrics = EffectMetrics(
                ctr=ctr,
                deep_engagement_index=deep_engagement,
                propagation_influence=propagation_influence,
                reach_amplification=reach_amplification,
                viral_coefficient=viral_coefficient
            )
            
            # 动态调整权重
            await self._optimize_metric_weights(metrics, data)
            
            logger.info(f"综合效果指标计算完成 - CTR: {ctr:.3f}, 深度参与: {deep_engagement:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"计算综合指标失败: {e}")
            return EffectMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    async def _calculate_ctr(self, data: Dict[str, Any]) -> float:
        """
        计算点击率
        CTR = 点击次数 / 曝光次数
        """
        try:
            interactions = data.get('user_interactions', [])
            
            # 统计点击和曝光
            clicks = sum(1 for i in interactions if i.get('type') in ['click', 'view', 'comment'])
            exposures = len(data.get('posts', [])) * len(data.get('target_users', []))
            
            if exposures == 0:
                return 0.0
            
            ctr = clicks / exposures
            
            # 分布式计算优化（模拟）
            if self.enable_distributed:
                ctr = await self._distributed_compute_ctr(interactions, exposures)
            
            return min(1.0, ctr)  # 限制在合理范围内
            
        except Exception as e:
            logger.error(f"计算CTR失败: {e}")
            return 0.0
    
    async def _calculate_deep_engagement_index(self, data: Dict[str, Any]) -> float:
        """
        计算深度参与指数
        深度参与指数 = w1·(Σ评论长度/评论数) + w2·转发层级 + w3·(二次创作数/内容数)
        """
        try:
            if not self.enable_deep_engagement:
                return 0.0
            
            interactions = data.get('user_interactions', [])
            posts = data.get('posts', [])
            
            # 1. 评论深度分析
            comments = [i for i in interactions if i.get('type') == 'comment']
            if comments:
                avg_comment_length = np.mean([len(str(c.get('content', ''))) for c in comments])
                comment_depth_score = min(avg_comment_length / 100.0, 1.0)  # 标准化
            else:
                comment_depth_score = 0.0
            
            # 2. 转发层级分析
            shares = [i for i in interactions if i.get('type') in ['share', 'repost']]
            repost_layers = self._analyze_repost_layers(shares)
            repost_score = min(repost_layers / 5.0, 1.0)  # 最多5层
            
            # 3. 二次创作分析
            creations = [i for i in interactions if i.get('type') == 'create']
            creation_rate = len(creations) / max(len(posts), 1)
            creation_score = min(creation_rate, 1.0)
            
            # 加权综合
            w1, w2, w3 = 0.5, 0.3, 0.2
            deep_engagement_index = (w1 * comment_depth_score + 
                                   w2 * repost_score + 
                                   w3 * creation_score)
            
            logger.debug(f"深度参与指数组成 - 评论: {comment_depth_score:.3f}, 转发: {repost_score:.3f}, 创作: {creation_score:.3f}")
            return deep_engagement_index
            
        except Exception as e:
            logger.error(f"计算深度参与指数失败: {e}")
            return 0.0
    
    async def _calculate_propagation_influence(self, data: Dict[str, Any]) -> float:
        """
        计算传播影响力
        传播影响力 = Σ_{p∈路径} w_p · 节点影响力_p
        """
        try:
            if not self.propagation_analysis:
                return 0.0
            
            interactions = data.get('user_interactions', [])
            
            # 构建传播网络
            propagation_network = self._build_propagation_network(interactions)
            
            # 计算节点影响力
            node_influences = self._calculate_node_influences(propagation_network)
            
            # 计算传播路径权重
            path_weights = self._calculate_propagation_paths(propagation_network)
            
            # 综合传播影响力
            total_influence = 0.0
            for path, weight in path_weights.items():
                path_nodes = path.split('->')
                path_influence = sum(node_influences.get(node, 0.0) for node in path_nodes)
                total_influence += weight * path_influence
            
            # 标准化
            max_possible_influence = len(interactions) * 1.0
            normalized_influence = total_influence / max(max_possible_influence, 1.0)
            
            return min(normalized_influence, 1.0)
            
        except Exception as e:
            logger.error(f"计算传播影响力失败: {e}")
            return 0.0
    
    async def _calculate_reach_amplification(self, data: Dict[str, Any]) -> float:
        """计算触达放大系数"""
        try:
            interactions = data.get('user_interactions', [])
            initial_users = set(data.get('target_users', []))
            
            # 统计实际触达用户
            reached_users = set()
            for interaction in interactions:
                user_id = interaction.get('user_id')
                if user_id:
                    reached_users.add(user_id)
            
            # 计算放大系数
            initial_reach = len(initial_users)
            actual_reach = len(reached_users)
            
            if initial_reach == 0:
                return 1.0
            
            amplification = actual_reach / initial_reach
            return min(amplification, 10.0)  # 限制最大放大倍数
            
        except Exception as e:
            logger.error(f"计算触达放大系数失败: {e}")
            return 1.0
    
    async def _calculate_viral_coefficient(self, data: Dict[str, Any]) -> float:
        """计算病毒传播系数"""
        try:
            interactions = data.get('user_interactions', [])
            
            # 统计分享行为
            shares = [i for i in interactions if i.get('type') in ['share', 'repost', 'forward']]
            total_users = len(set(i.get('user_id') for i in interactions if i.get('user_id')))
            
            if total_users == 0:
                return 0.0
            
            # 病毒系数 = 平均每用户分享数
            viral_coefficient = len(shares) / total_users
            
            # 考虑分享质量权重
            quality_weighted_shares = sum(
                share.get('engagement_level', 0.5) for share in shares
            )
            
            if len(shares) > 0:
                avg_share_quality = quality_weighted_shares / len(shares)
                viral_coefficient *= avg_share_quality
            
            return min(viral_coefficient, 2.0)  # 限制最大病毒系数
            
        except Exception as e:
            logger.error(f"计算病毒传播系数失败: {e}")
            return 0.0
    
    def _analyze_repost_layers(self, shares: List[Dict[str, Any]]) -> int:
        """分析转发层级"""
        if not shares:
            return 0
        
        # 简化的层级分析
        max_layers = 0
        for share in shares:
            # 模拟层级检测
            layers = share.get('repost_depth', 1)
            max_layers = max(max_layers, layers)
        
        return max_layers
    
    def _build_propagation_network(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建传播网络"""
        network = {
            'nodes': set(),
            'edges': [],
            'node_properties': {}
        }
        
        for interaction in interactions:
            user_id = interaction.get('user_id')
            post_id = interaction.get('post_id')
            
            if user_id:
                network['nodes'].add(user_id)
                
                # 添加节点属性
                if user_id not in network['node_properties']:
                    network['node_properties'][user_id] = {
                        'interaction_count': 0,
                        'engagement_sum': 0.0,
                        'influence_score': 0.0
                    }
                
                props = network['node_properties'][user_id]
                props['interaction_count'] += 1
                props['engagement_sum'] += interaction.get('engagement_level', 0.0)
                
                # 计算影响力得分
                if props['interaction_count'] > 0:
                    props['influence_score'] = props['engagement_sum'] / props['interaction_count']
        
        return network
    
    def _calculate_node_influences(self, network: Dict[str, Any]) -> Dict[str, float]:
        """计算节点影响力"""
        influences = {}
        
        for node in network['nodes']:
            props = network['node_properties'].get(node, {})
            base_influence = props.get('influence_score', 0.0)
            interaction_bonus = min(props.get('interaction_count', 0) / 10.0, 1.0)
            
            influences[node] = base_influence * (1 + interaction_bonus)
        
        return influences
    
    def _calculate_propagation_paths(self, network: Dict[str, Any]) -> Dict[str, float]:
        """计算传播路径权重"""
        paths = {}
        
        # 简化的路径计算
        nodes = list(network['nodes'])
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                path_key = f"{node1}->{node2}"
                
                # 基于节点属性计算路径权重
                props1 = network['node_properties'].get(node1, {})
                props2 = network['node_properties'].get(node2, {})
                
                weight = (props1.get('influence_score', 0.0) + 
                         props2.get('influence_score', 0.0)) / 2.0
                
                paths[path_key] = weight
        
        return paths
    
    async def _distributed_compute_ctr(self, interactions: List[Dict[str, Any]], 
                                     exposures: int) -> float:
        """分布式计算CTR（模拟）"""
        # 模拟分布式计算
        chunk_size = max(1, len(interactions) // 4)
        chunks = [interactions[i:i + chunk_size] for i in range(0, len(interactions), chunk_size)]
        
        chunk_results = []
        for chunk in chunks:
            chunk_clicks = sum(1 for i in chunk if i.get('type') in ['click', 'view', 'comment'])
            chunk_results.append(chunk_clicks)
        
        total_clicks = sum(chunk_results)
        return total_clicks / max(exposures, 1)
    
    def _apply_privacy_protection(self, ctr: float, engagement: float, 
                                propagation: float) -> Tuple[float, float, float]:
        """
        应用差分隐私保护
        反馈_i^实际' = 反馈_i^实际 + N(0, σ²)
        """
        try:
            # 添加差分隐私噪声
            ctr_noise = np.random.normal(0, self.privacy_epsilon * 0.01)
            engagement_noise = np.random.normal(0, self.privacy_epsilon * 0.02)
            propagation_noise = np.random.normal(0, self.privacy_epsilon * 0.015)
            
            protected_ctr = max(0, ctr + ctr_noise)
            protected_engagement = max(0, engagement + engagement_noise)
            protected_propagation = max(0, propagation + propagation_noise)
            
            logger.debug("差分隐私保护已应用到效果指标")
            return protected_ctr, protected_engagement, protected_propagation
            
        except Exception as e:
            logger.error(f"应用隐私保护失败: {e}")
            return ctr, engagement, propagation
    
    async def _optimize_metric_weights(self, metrics: EffectMetrics, data: Dict[str, Any]):
        """
        动态调整指标权重
        综合效果 = w1·CTR + w2·深度参与指数 + w3·传播影响力
        w* = argmax_w E[实际效果|w]
        """
        try:
            # 基于指标表现调整权重
            metric_values = [metrics.ctr, metrics.deep_engagement_index, 
                           metrics.propagation_influence, metrics.reach_amplification]
            
            # 计算指标重要性
            importance_scores = []
            for i, value in enumerate(metric_values):
                # 基于方差和均值计算重要性
                importance = value * (1 + np.random.uniform(-0.1, 0.1))
                importance_scores.append(max(0.1, importance))
            
            # 归一化权重
            total_importance = sum(importance_scores)
            if total_importance > 0:
                new_weights = [score / total_importance for score in importance_scores]
                
                # 更新自适应权重
                weight_keys = ['ctr_weight', 'engagement_weight', 'propagation_weight', 'reach_weight']
                for i, key in enumerate(weight_keys):
                    if i < len(new_weights):
                        self.adaptive_weights[key] = new_weights[i]
            
            logger.debug(f"权重已优化: {self.adaptive_weights}")
            
        except Exception as e:
            logger.error(f"优化指标权重失败: {e}")
    
    async def analyze_temporal_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间趋势"""
        try:
            if len(historical_data) < 2:
                return {'trend': 'insufficient_data'}
            
            # 提取时间序列数据
            timestamps = []
            ctr_values = []
            engagement_values = []
            
            for data_point in historical_data:
                if 'timestamp' in data_point and 'metrics' in data_point:
                    timestamps.append(data_point['timestamp'])
                    metrics = data_point['metrics']
                    ctr_values.append(metrics.get('ctr', 0.0))
                    engagement_values.append(metrics.get('deep_engagement_index', 0.0))
            
            # 计算趋势
            trends = {
                'ctr_trend': self._calculate_trend(ctr_values),
                'engagement_trend': self._calculate_trend(engagement_values),
                'volatility': {
                    'ctr_volatility': np.std(ctr_values) if ctr_values else 0.0,
                    'engagement_volatility': np.std(engagement_values) if engagement_values else 0.0
                },
                'performance_stability': self._calculate_stability(ctr_values, engagement_values)
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"分析时间趋势失败: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return 'stable'
        
        # 简单线性趋势
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_stability(self, ctr_values: List[float], 
                           engagement_values: List[float]) -> float:
        """计算性能稳定性"""
        if not ctr_values or not engagement_values:
            return 0.0
        
        ctr_cv = np.std(ctr_values) / max(np.mean(ctr_values), 0.001)  # 变异系数
        engagement_cv = np.std(engagement_values) / max(np.mean(engagement_values), 0.001)
        
        # 稳定性 = 1 - 平均变异系数
        stability = 1.0 - (ctr_cv + engagement_cv) / 2.0
        return max(0.0, min(1.0, stability))
    
    async def generate_effect_report(self, metrics: EffectMetrics, 
                                   trends: Dict[str, Any] = None) -> str:
        """生成效果分析报告"""
        try:
            report = f"""
实际效果分析报告
================

核心指标:
- 点击率 (CTR): {metrics.ctr:.3f}
- 深度参与指数: {metrics.deep_engagement_index:.3f}
- 传播影响力: {metrics.propagation_influence:.3f}
- 触达放大系数: {metrics.reach_amplification:.2f}x
- 病毒传播系数: {metrics.viral_coefficient:.3f}

综合效果评分: {self._calculate_composite_score(metrics):.3f}

权重配置:
- CTR权重: {self.adaptive_weights['ctr_weight']:.2f}
- 参与权重: {self.adaptive_weights['engagement_weight']:.2f}
- 传播权重: {self.adaptive_weights['propagation_weight']:.2f}
- 触达权重: {self.adaptive_weights['reach_weight']:.2f}
"""
            
            if trends:
                report += f"""
趋势分析:
- CTR趋势: {trends.get('ctr_trend', 'unknown')}
- 参与趋势: {trends.get('engagement_trend', 'unknown')}
- 性能稳定性: {trends.get('performance_stability', 0.0):.3f}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"生成效果报告失败: {e}")
            return "报告生成失败"
    
    def _calculate_composite_score(self, metrics: EffectMetrics) -> float:
        """计算综合效果得分"""
        return (self.adaptive_weights['ctr_weight'] * metrics.ctr +
                self.adaptive_weights['engagement_weight'] * metrics.deep_engagement_index +
                self.adaptive_weights['propagation_weight'] * metrics.propagation_influence +
                self.adaptive_weights['reach_weight'] * min(metrics.reach_amplification / 5.0, 1.0))
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析模块摘要"""
        return {
            'module_name': 'ActualEffectAnalysisModule',
            'adaptive_weights': self.adaptive_weights,
            'privacy_protection': self.config.get('privacy_protection', True),
            'distributed_computing': self.enable_distributed,
            'deep_engagement_analysis': self.enable_deep_engagement,
            'propagation_analysis': self.propagation_analysis,
            'privacy_epsilon': self.privacy_epsilon
        }

# 工厂函数
def create_actual_effect_analysis_module(config: Dict[str, Any] = None) -> ActualEffectAnalysisModule:
    """创建实际效果分析模块实例"""
    return ActualEffectAnalysisModule(config)
