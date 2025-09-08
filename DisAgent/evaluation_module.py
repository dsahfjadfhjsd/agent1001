# -*- coding: utf-8 -*-
"""
DISTAgent 评估反馈模块
量化用户认知反馈，动态优化分发策略
"""

import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# 检查依赖可用性
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch不可用，参数微调功能受限")

KAFKA_AVAILABLE = False  # 完全移除Kafka依赖，改用本地内存反馈处理


@dataclass
class FeedbackMetrics:
    """反馈指标"""
    ctr: float  # 点击通过率
    engagement_depth: float  # 深度参与指数
    cognitive_consistency: float  # 认知一致性得分
    sentiment_alignment: float  # 情感极性强度
    viral_coefficient: float  # 传播系数
    user_satisfaction: float  # 用户满意度
    constructive_engagement: float = 0.0  # 构造性参与度（新增，向后兼容默认0）


@dataclass
class DistributionFeedback:
    """分发反馈"""
    distribution_id: str
    content_id: str
    user_feedbacks: List[Dict[str, Any]]
    metrics: FeedbackMetrics
    timestamp: datetime
    platform: str


class MultiDimensionalMetricsCalculator:
    """多维度指标计算器"""
    
    def __init__(self):
        self.metric_weights = {
            'ctr': 0.3,
            'engagement_depth': 0.25,
            'cognitive_consistency': 0.2,
            'sentiment_alignment': 0.15,
            'viral_coefficient': 0.05,
            'constructive_engagement': 0.05,
        }
    
    def calculate_ctr(self, clicks: int, impressions: int) -> float:
        """计算点击通过率"""
        if impressions == 0:
            return 0.0
        return clicks / impressions
    
    def calculate_engagement_depth(self, interactions: List[Dict[str, Any]]) -> float:
        """计算深度参与指数"""
        if not interactions:
            return 0.0
        
        total_score = 0.0
        for interaction in interactions:
            # 评论长度权重
            comment_length = len(interaction.get('comment', ''))
            comment_score = min(1.0, comment_length / 100.0) * 0.4
            
            # 转发层级权重
            repost_level = interaction.get('repost_level', 0)
            repost_score = min(1.0, repost_level / 5.0) * 0.3
            
            # 二次创作权重
            creation_count = interaction.get('secondary_creation', 0)
            creation_score = min(1.0, creation_count / 3.0) * 0.3
            
            total_score += comment_score + repost_score + creation_score
        
        return total_score / len(interactions)
    
    def calculate_cognitive_consistency(self, target_attributes: Dict[str, Any], 
                                     user_responses: List[Dict[str, Any]]) -> float:
        """计算认知一致性得分"""
        if not user_responses:
            return 0.0
        
        consistency_scores = []
        
        for response in user_responses:
            response_sentiment = response.get('sentiment', 'neutral')
            target_sentiment = target_attributes.get('emotion', 'neutral')
            
            # 情感一致性
            if response_sentiment == target_sentiment:
                sentiment_consistency = 1.0
            elif (response_sentiment in ['positive', 'negative'] and 
                  target_sentiment in ['positive', 'negative']):
                sentiment_consistency = 0.5
            else:
                sentiment_consistency = 0.0
            
            # 立场一致性
            response_stance = response.get('stance', 'neutral')
            target_stance = target_attributes.get('stance', 'neutral')
            
            if response_stance == target_stance:
                stance_consistency = 1.0
            else:
                stance_consistency = 0.0
            
            # 综合一致性
            consistency = (sentiment_consistency + stance_consistency) / 2.0
            consistency_scores.append(consistency)
        
        return sum(consistency_scores) / len(consistency_scores)
    
    def calculate_sentiment_alignment(self, content_sentiment: str, 
                                    user_sentiments: List[str]) -> float:
        """计算情感极性强度"""
        if not user_sentiments:
            return 0.0
        
        alignment_count = sum(1 for sentiment in user_sentiments 
                            if sentiment == content_sentiment)
        
        return alignment_count / len(user_sentiments)
    
    def calculate_viral_coefficient(self, shares: int, original_reach: int) -> float:
        """计算传播系数"""
        if original_reach == 0:
            return 0.0
        
        return min(5.0, shares / original_reach)  # 限制最大传播系数为5
    
    def calculate_composite_score(self, metrics: FeedbackMetrics) -> float:
        """计算综合评分"""
        score = (
            metrics.ctr * self.metric_weights['ctr'] +
            metrics.engagement_depth * self.metric_weights['engagement_depth'] +
            metrics.cognitive_consistency * self.metric_weights['cognitive_consistency'] +
            metrics.sentiment_alignment * self.metric_weights['sentiment_alignment'] +
            (metrics.viral_coefficient / 5.0) * self.metric_weights['viral_coefficient'] +
            metrics.constructive_engagement * self.metric_weights['constructive_engagement']
        )
        return min(1.0, score)

    def calculate_constructive_engagement(self, 
                                          interactions: List[Dict[str, Any]], 
                                          user_responses: List[Dict[str, Any]]) -> float:
        """计算构造性参与度
        简单启发式：
        - 正向关键词（建议/理性/证据/来源/数据/分析/改进/感谢/请教/讨论/参考）加分
        - 负向/攻击性词（骂/喷/垃圾/滚/闭嘴/攻击/辱骂/人身攻击）减分
        - 提问符号和字数适中加分
        最终返回 [0,1] 范围的平均分。
        """
        if not interactions:
            return 0.0
        pos_kw = ["建议", "理性", "证据", "来源", "数据", "分析", "改进", "感谢", "请教", "讨论", "参考"]
        neg_kw = ["骂", "喷", "垃圾", "滚", "闭嘴", "攻击", "辱骂", "人身攻击"]

        scores: List[float] = []
        for i, inter in enumerate(interactions):
            text = str(inter.get('comment', '') or '')
            s = 0.0
            # 基础：长度适中（20-200）加分
            L = len(text)
            if 20 <= L <= 200:
                s += 0.3
            elif L > 200:
                s += 0.2
            # 问句与礼貌标记
            if '?' in text or '？' in text:
                s += 0.15
            if any(k in text for k in ["请", "谢谢", "感谢", "麻烦"]):
                s += 0.1
            # 关键词
            if any(k in text for k in pos_kw):
                s += 0.3
            if any(k in text for k in neg_kw):
                s -= 0.4
            # 结合用户响应情感（负面强惩罚）
            resp = user_responses[i] if i < len(user_responses) else {}
            sentiment = str(resp.get('sentiment', 'neutral')).lower()
            if sentiment == 'negative':
                s -= 0.1
            elif sentiment == 'positive':
                s += 0.05

            scores.append(max(0.0, min(1.0, s)))

        return sum(scores) / len(scores)


class AdaptiveParameterOptimizer:
    """自适应参数优化器"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.parameter_history = []
        
        if TORCH_AVAILABLE:
            self.optimizer_net = self._build_optimizer_network()
            self.optimizer = optim.Adam(self.optimizer_net.parameters(), lr=learning_rate)
    
    def _build_optimizer_network(self):
        """构建优化网络"""
        if not TORCH_AVAILABLE:
            return None
        
        return nn.Sequential(
            nn.Linear(32, 64),  # 输入：当前参数 + 反馈指标
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  # 输出：参数调整建议
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def optimize_parameters(self, current_params: Dict[str, float], 
                          feedback_metrics: FeedbackMetrics) -> Dict[str, float]:
        """优化参数"""
        if TORCH_AVAILABLE and self.optimizer_net:
            return self._neural_optimization(current_params, feedback_metrics)
        else:
            return self._rule_based_optimization(current_params, feedback_metrics)
    
    def _neural_optimization(self, current_params: Dict[str, float], 
                           feedback_metrics: FeedbackMetrics) -> Dict[str, float]:
        """神经网络优化"""
        # 构建输入向量
        param_vector = self._params_to_vector(current_params)
        metrics_vector = self._metrics_to_vector(feedback_metrics)
        
        if not TORCH_AVAILABLE or self.optimizer_net is None:
            # 如果torch不可用，使用简单的规则优化
            return self._simple_rule_optimization(current_params, feedback_metrics)
        
        try:
            input_vector = torch.cat([param_vector, metrics_vector])
            
            # 网络预测
            with torch.no_grad():
                adjustments = self.optimizer_net(input_vector.unsqueeze(0))[0]
        except NameError:
            return self._simple_rule_optimization(current_params, feedback_metrics)
        
        # 应用调整
        optimized_params = {}
        param_keys = ['posts_per_round', 'users_per_post', 'hot_post_ratio', 
                     'personalization_strength']
        
        for i, key in enumerate(param_keys[:len(adjustments)]):
            if key in current_params:
                adjustment = adjustments[i].item() * 0.1  # 限制调整幅度
                optimized_params[key] = max(0.1, min(1.0, 
                    current_params[key] + adjustment))
        
        return optimized_params
    
    def _rule_based_optimization(self, current_params: Dict[str, float], 
                               feedback_metrics: FeedbackMetrics) -> Dict[str, float]:
        """基于规则的优化"""
        optimized_params = current_params.copy()
        
        # CTR优化
        if feedback_metrics.ctr < 0.05:
            # CTR过低，增加个性化强度
            optimized_params['personalization_strength'] = min(1.0,
                current_params.get('personalization_strength', 0.5) + 0.1)
        elif feedback_metrics.ctr > 0.2:
            # CTR过高，可以适度降低个性化强度
            optimized_params['personalization_strength'] = max(0.1,
                current_params.get('personalization_strength', 0.5) - 0.05)
        
        # 参与度优化
        if feedback_metrics.engagement_depth < 0.3:
            # 参与度低，增加每帖用户数
            optimized_params['users_per_post'] = min(20,
                current_params.get('users_per_post', 10) + 2)
        
        # 认知一致性优化
        if feedback_metrics.cognitive_consistency < 0.5:
            # 一致性低，调整热门帖子比例
            optimized_params['hot_post_ratio'] = max(0.2,
                current_params.get('hot_post_ratio', 0.4) - 0.1)
        
        return optimized_params
    
    def _simple_rule_optimization(self, current_params: Dict[str, float], 
                                feedback_metrics: FeedbackMetrics) -> Dict[str, float]:
        """简单规则优化（torch不可用时的fallback）"""
        optimized_params = current_params.copy()
        
        # CTR优化
        if feedback_metrics.ctr < 0.05:
            optimized_params['posts_per_round'] = min(10, 
                current_params.get('posts_per_round', 5) + 1)
        
        # 参与度优化
        if feedback_metrics.engagement_depth < 0.3:
            optimized_params['users_per_post'] = max(5,
                current_params.get('users_per_post', 10) - 1)
        
        # 认知一致性优化
        if feedback_metrics.cognitive_consistency < 0.5:
            optimized_params['hot_post_ratio'] = max(0.2,
                current_params.get('hot_post_ratio', 0.4) - 0.05)
        
        return optimized_params
    
    def _params_to_vector(self, params: Dict[str, float]):
        """参数转向量"""
        if not TORCH_AVAILABLE:
            # 如果torch不可用，返回numpy数组
            import numpy as np
            vector = np.zeros(16)
            param_keys = ['posts_per_round', 'users_per_post', 'hot_post_ratio', 
                         'personalization_strength']
            
            for i, key in enumerate(param_keys):
                if key in params:
                    vector[i] = params[key]
            return vector
        
        vector = torch.zeros(16)
        param_keys = ['posts_per_round', 'users_per_post', 'hot_post_ratio', 
                     'personalization_strength']
        
        for i, key in enumerate(param_keys):
            if key in params:
                vector[i] = params[key]
        
        return vector
    
    def _metrics_to_vector(self, metrics: FeedbackMetrics):
        """指标转向量"""
        if not TORCH_AVAILABLE:
            import numpy as np
            return np.array([
                metrics.ctr,
                metrics.engagement_depth,
                metrics.cognitive_consistency,
            ])
        
        return torch.tensor([
            metrics.ctr,
            metrics.engagement_depth,
            metrics.cognitive_consistency,
            metrics.sentiment_alignment,
            metrics.viral_coefficient / 5.0,  # 归一化
            metrics.user_satisfaction
        ]).float()


class RealTimeFeedbackProcessor:
    """实时反馈处理器"""
    
    def __init__(self):
        self.feedback_queue = asyncio.Queue() if asyncio else []
    
    async def initialize(self):
        """异步初始化方法"""
        try:
            logger.info("开始初始化评估反馈模块...")
            
            # 初始化优化网络
            if TORCH_AVAILABLE:
                logger.info("初始化神经网络优化器...")
            else:
                logger.info("PyTorch不可用，使用规则优化")
            
            # 初始化反馈处理组件
            logger.info("初始化实时反馈处理组件...")
            
            logger.info("评估反馈模块初始化完成")
            
        except Exception as e:
            logger.error(f"评估反馈模块初始化失败: {e}")
            logger.warning("使用降级模式：简单规则优化")
    
    async def collect_feedback(self, distribution_id: str, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """按分发ID收集反馈（内存队列）"""
        items: List[Dict[str, Any]] = []
        try:
            while True:
                try:
                    feedback = await asyncio.wait_for(self.feedback_queue.get(), timeout=timeout)
                    if feedback and feedback.get('distribution_id') == distribution_id:
                        items.append(feedback)
                except asyncio.TimeoutError:
                    break
        except Exception as e:
            logger.error(f"收集反馈失败: {e}")
        return items
    
    async def publish_feedback(self, feedback: Dict[str, Any]):
        """发布反馈"""
        try:
            if asyncio:
                await self.feedback_queue.put(feedback)
            
            logger.debug(f"发布反馈: {feedback.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"发布反馈失败: {e}")


class EvaluationModule:
    """评估反馈模块主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化组件
        self.metrics_calculator = MultiDimensionalMetricsCalculator()
        self.parameter_optimizer = AdaptiveParameterOptimizer(
            learning_rate=self.config.get('learning_rate', 0.01)
        )
        self.feedback_processor = RealTimeFeedbackProcessor()
        
        # 评估历史
        self.evaluation_history = []
        
        # 权重更新策略
        self.weight_update_strategy = self.config.get('weight_strategy', 'adaptive')
        
        logger.info("评估反馈模块初始化完成")
    
    async def evaluate_distribution_performance(self, 
                                              distribution_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估分发性能"""
        try:
            distribution_id = distribution_data['distribution_id']
            
            # 收集实时反馈（内存队列）
            feedback_items = await self.feedback_processor.collect_feedback(distribution_id)
            
            # 计算多维指标
            metrics = self._calculate_distribution_metrics(distribution_data, feedback_items)
            
            # 生成评估报告
            evaluation_report = {
                'distribution_id': distribution_id,
                'metrics': metrics,
                'feedback_count': len(feedback_items),
                'evaluation_time': datetime.now().isoformat(),
                'composite_score': self.metrics_calculator.calculate_composite_score(metrics)
            }
            
            # 记录评估历史
            self.evaluation_history.append(evaluation_report)
            
            return evaluation_report
            
        except Exception as e:
            logger.error(f"评估分发性能失败: {e}")
            return {'error': str(e)}
    
    def _calculate_distribution_metrics(self, distribution_data: Dict[str, Any], 
                                      feedback_items: List[Dict[str, Any]]) -> FeedbackMetrics:
        """计算分发指标"""
        # 提取统计数据
        clicks = sum(item.get('clicks', 0) for item in feedback_items)
        impressions = sum(item.get('impressions', 0) for item in feedback_items)
        interactions = [item.get('interaction', {}) for item in feedback_items]
        user_responses = [item.get('user_response', {}) for item in feedback_items]
        shares = sum(item.get('shares', 0) for item in feedback_items)
        
        # 计算各项指标
        ctr = self.metrics_calculator.calculate_ctr(clicks, impressions)
        
        engagement_depth = self.metrics_calculator.calculate_engagement_depth(interactions)
        
        target_attributes = distribution_data.get('target_attributes', {})
        cognitive_consistency = self.metrics_calculator.calculate_cognitive_consistency(
            target_attributes, user_responses
        )
        
        content_sentiment = distribution_data.get('content_sentiment', 'neutral')
        user_sentiments = [resp.get('sentiment', 'neutral') for resp in user_responses]
        sentiment_alignment = self.metrics_calculator.calculate_sentiment_alignment(
            content_sentiment, user_sentiments
        )
        
        viral_coefficient = self.metrics_calculator.calculate_viral_coefficient(
            shares, impressions
        )
        
        # 用户满意度（基于反馈评分）
        satisfaction_scores = [item.get('satisfaction', 0.5) for item in feedback_items]
        user_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.5
        
        # 构造性参与度
        constructive_engagement = self.metrics_calculator.calculate_constructive_engagement(
            interactions, user_responses
        )
        
        return FeedbackMetrics(
            ctr=ctr,
            engagement_depth=engagement_depth,
            cognitive_consistency=cognitive_consistency,
            sentiment_alignment=sentiment_alignment,
            constructive_engagement=constructive_engagement,
            viral_coefficient=viral_coefficient,
            user_satisfaction=user_satisfaction
        )

    # ... (rest of the code remains the same)

    async def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        try:
            return {
                'module_name': 'EvaluationModule',
                'adaptive_metrics_ready': hasattr(self, 'adaptive_metrics') and self.adaptive_metrics is not None,
                'feedback_processor_ready': hasattr(self, 'feedback_processor') and self.feedback_processor is not None,
                'metrics_calculator_ready': hasattr(self, 'metrics_calculator') and self.metrics_calculator is not None,
                'evaluation_history_count': len(self.evaluation_history) if hasattr(self, 'evaluation_history') else 0,
                'multi_dimensional_evaluation': self.enable_multi_dimensional,
                'adaptive_optimization': self.enable_adaptive_optimization,
                'status': 'healthy'
            }
        except Exception as e:
            return {
                'module_name': 'EvaluationModule',
                'status': 'error',
                'error': str(e)
            }

    # ... (rest of the code remains the same)

    async def optimize_distribution_strategy(self, current_params: Dict[str, float], 
                                           recent_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """优化分发策略"""
        try:
            if not recent_evaluations:
                return current_params
            
            # 计算平均反馈指标
            avg_metrics = self._calculate_average_metrics(recent_evaluations)
            
            # 优化参数
            optimized_params = self.parameter_optimizer.optimize_parameters(
                current_params, avg_metrics
            )
            
            # 动态权重调整
            if self.weight_update_strategy == 'adaptive':
                self._update_metric_weights(avg_metrics)
            
            # 生成优化报告
            optimization_report = {
                'original_params': current_params,
                'optimized_params': optimized_params,
                'improvement_metrics': avg_metrics,
                'optimization_timestamp': datetime.now().isoformat(),
                'weight_updates': self.metrics_calculator.metric_weights
            }
            
            return optimization_report
            
        except Exception as e:
            logger.error(f"优化分发策略失败: {e}")
            return {'error': str(e)}
    
    def _calculate_average_metrics(self, evaluations: List[Dict[str, Any]]) -> FeedbackMetrics:
        """计算平均指标"""
        if not evaluations:
            return FeedbackMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        total_ctr = sum(eval_data['metrics'].ctr for eval_data in evaluations)
        total_engagement = sum(eval_data['metrics'].engagement_depth for eval_data in evaluations)
        total_consistency = sum(eval_data['metrics'].cognitive_consistency for eval_data in evaluations)
        total_alignment = sum(eval_data['metrics'].sentiment_alignment for eval_data in evaluations)
        total_viral = sum(eval_data['metrics'].viral_coefficient for eval_data in evaluations)
        total_satisfaction = sum(eval_data['metrics'].user_satisfaction for eval_data in evaluations)
        
        count = len(evaluations)
        
        return FeedbackMetrics(
            ctr=total_ctr / count,
            engagement_depth=total_engagement / count,
            cognitive_consistency=total_consistency / count,
            sentiment_alignment=total_alignment / count,
            viral_coefficient=total_viral / count,
            user_satisfaction=total_satisfaction / count
        )
    
    def _update_metric_weights(self, metrics: FeedbackMetrics):
        """更新指标权重"""
        # 基于性能动态调整权重
        performance_scores = {
            'ctr': metrics.ctr,
            'engagement_depth': metrics.engagement_depth,
            'cognitive_consistency': metrics.cognitive_consistency,
            'sentiment_alignment': metrics.sentiment_alignment,
            'viral_coefficient': metrics.viral_coefficient / 5.0
        }
        
        # 提升表现好的指标权重，降低表现差的指标权重
        total_adjustment = 0.0
        for metric, score in performance_scores.items():
            if metric in self.metrics_calculator.metric_weights:
                if score > 0.7:  # 高性能指标
                    adjustment = 0.05
                elif score < 0.3:  # 低性能指标
                    adjustment = -0.05
                else:
                    adjustment = 0.0
                
                self.metrics_calculator.metric_weights[metric] += adjustment
                total_adjustment += adjustment
        
        # 归一化权重
        total_weight = sum(self.metrics_calculator.metric_weights.values())
        if total_weight > 0:
            for metric in self.metrics_calculator.metric_weights:
                self.metrics_calculator.metric_weights[metric] /= total_weight
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评估统计"""
        if not self.evaluation_history:
            return {'message': '暂无评估历史'}
        
        recent_evaluations = self.evaluation_history[-10:]  # 最近10次评估
        
        avg_composite_score = sum(eval_data['composite_score'] 
                                for eval_data in recent_evaluations) / len(recent_evaluations)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'recent_average_score': avg_composite_score,
            'current_weights': self.metrics_calculator.metric_weights,
            'last_evaluation': self.evaluation_history[-1]['evaluation_time'],
            'performance_trend': self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """计算性能趋势"""
        if len(self.evaluation_history) < 2:
            return 'insufficient_data'
        
        recent_scores = [eval_data['composite_score'] 
                        for eval_data in self.evaluation_history[-5:]]
        
        if len(recent_scores) >= 3:
            # 简单的趋势判断
            early_avg = sum(recent_scores[:2]) / 2
            late_avg = sum(recent_scores[-2:]) / 2
            
            if late_avg > early_avg + 0.05:
                return 'improving'
            elif late_avg < early_avg - 0.05:
                return 'declining'
            else:
                return 'stable'
        
        return 'stable'


# 工厂函数
def create_evaluation_module(config: Dict[str, Any] = None) -> EvaluationModule:
    """创建评估反馈模块实例"""
    return EvaluationModule(config)


# 使用示例
if __name__ == "__main__":
    # 创建评估模块
    eval_module = create_evaluation_module({
        'learning_rate': 0.01,
        'weight_strategy': 'adaptive'
    })
    
    async def main():
        # 示例分发数据
        distribution_data = {
            'distribution_id': 'dist_001',
            'content_sentiment': 'positive',
            'target_attributes': {
                'emotion': 'positive',
                'stance': 'supportive'
            }
        }
        
        # 模拟反馈数据
        await eval_module.feedback_processor.publish_feedback({
            'distribution_id': 'dist_001',
            'clicks': 50,
            'impressions': 1000,
            'shares': 10,
            'satisfaction': 0.8,
            'user_response': {
                'sentiment': 'positive',
                'stance': 'supportive'
            },
            'interaction': {
                'comment': '这个内容很有意思！',
                'repost_level': 2,
                'secondary_creation': 1
            }
        })
        
        # 评估性能
        evaluation_result = await eval_module.evaluate_distribution_performance(distribution_data)
        print(f"评估结果: {evaluation_result}")
        
        # 优化策略
        current_params = {
            'posts_per_round': 5,
            'users_per_post': 10,
            'hot_post_ratio': 0.4,
            'personalization_strength': 0.7
        }
        
        optimization_result = await eval_module.optimize_distribution_strategy(
            current_params, [evaluation_result]
        )
        print(f"优化结果: {optimization_result}")
        
        # 获取统计信息
        stats = eval_module.get_evaluation_statistics()
        print(f"评估统计: {stats}")
    
    # 运行示例
    asyncio.run(main())
