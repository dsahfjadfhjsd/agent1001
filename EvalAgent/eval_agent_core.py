# -*- coding: utf-8 -*-
"""
EvalAgent 核心类
整合5个模块的主要评估智能体
"""

import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .data_collection_module import DataCollectionModule, create_data_collection_module
from .simulation_environment_module import SimulationEnvironmentModule, create_simulation_environment_module
from .actual_effect_analysis_module import ActualEffectAnalysisModule, create_actual_effect_analysis_module
from .cognitive_impact_assessment_module import CognitiveImpactAssessmentModule, create_cognitive_impact_assessment_module
from .optimization_feedback_module import OptimizationFeedbackModule, create_optimization_feedback_module

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    ctr: float  # 点击率
    deep_engagement_index: float  # 深度参与指数
    propagation_influence: float  # 传播影响力
    cognitive_consistency: float  # 认知一致性
    sentiment_polarity: float  # 情感极性强度
    behavior_conversion_rate: float  # 行为转化率
    
@dataclass
class DistributionExpectation:
    """分发期望数据类"""
    target_ctr: float = 0.1
    target_engagement: float = 0.7
    target_consistency: float = 0.8
    target_conversion: float = 0.15
    target_propagation: float = 0.6

class EvalAgent:
    """
    EvalAgent 评估智能体核心类
    整合5个核心模块：数据收集、模拟环境、实际效果分析、认知影响评估、优化反馈
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化各个模块
        self.data_collection = create_data_collection_module(
            self.config.get('data_collection', {})
        )
        self.simulation_environment = create_simulation_environment_module(
            self.config.get('simulation_environment', {})
        )
        self.effect_analysis = create_actual_effect_analysis_module(
            self.config.get('effect_analysis', {})
        )
        self.cognitive_assessment = create_cognitive_impact_assessment_module(
            self.config.get('cognitive_assessment', {})
        )
        self.optimization_feedback = create_optimization_feedback_module(
            self.config.get('optimization_feedback', {})
        )
        
        # 默认分发期望
        self.distribution_expectation = DistributionExpectation()
        
        # 评估历史
        self.evaluation_history = []
        
        logger.info("EvalAgent 评估智能体初始化完成")
    
    async def evaluate_distribution_performance(self, 
                                              simulation_results: Dict[str, Any],
                                              actual_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        评估分发性能 - EvalAgent工作流程
        """
        try:
            logger.info("开始EvalAgent分发性能评估...")
            
            # 步骤1: 数据收集
            logger.info("步骤1: 数据收集")
            collected_simulation_data = await self.data_collection.collect_simulation_data(
                simulation_results
            )
            
            collected_actual_data = {}
            if actual_data:
                collected_actual_data = actual_data
            else:
                # 收集实际数据
                collected_actual_data = await self.data_collection.collect_actual_data()
            
            # 步骤2: 模拟环境运行（如果需要额外模拟）
            logger.info("步骤2: 模拟环境分析")
            simulation_analysis = await self._analyze_simulation_environment(
                collected_simulation_data
            )
            
            # 步骤3: 实际效果分析
            logger.info("步骤3: 实际效果分析")
            effect_metrics = await self.effect_analysis.calculate_comprehensive_metrics(
                collected_simulation_data
            )
            
            # 步骤4: 认知影响评估
            logger.info("步骤4: 认知影响评估")
            cognitive_metrics = await self.cognitive_assessment.assess_comprehensive_cognitive_impact(
                content_data=simulation_results.get('content_data', {}),
                user_interactions=collected_simulation_data.get('user_interactions', []),
                target_attributes={'target_stance': 'positive', 'target_sentiment': 0.7}
            )
            
            # 步骤5: 优化反馈
            logger.info("步骤5: 优化反馈生成")
            current_metrics = {
                'ctr': effect_metrics.ctr,
                'deep_engagement_index': effect_metrics.deep_engagement_index,
                'propagation_influence': effect_metrics.propagation_influence,
                'cognitive_consistency': cognitive_metrics.consistency_score,
                'sentiment_polarity_strength': cognitive_metrics.sentiment_polarity_strength,
                'behavior_conversion_rate': cognitive_metrics.behavior_conversion_rate
            }
            
            optimization_result = await self.optimization_feedback.optimize_distribution_strategy(
                current_metrics=current_metrics,
                expectation=self._expectation_to_dict(),
                historical_data=self.evaluation_history[-10:] if self.evaluation_history else None
            )
            
            # 综合评估结果
            evaluation_result = {
                'timestamp': datetime.now().isoformat(),
                'evaluation_id': f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'data_collection': {
                    'simulation_data_summary': self.data_collection.get_collected_data_summary(),
                    'data_quality_score': collected_simulation_data.get('metadata', {}).get('data_quality_score', 0.5)
                },
                'simulation_analysis': simulation_analysis,
                'effect_metrics': {
                    'ctr': effect_metrics.ctr,
                    'deep_engagement_index': effect_metrics.deep_engagement_index,
                    'propagation_influence': effect_metrics.propagation_influence,
                    'reach_amplification': effect_metrics.reach_amplification,
                    'viral_coefficient': effect_metrics.viral_coefficient
                },
                'cognitive_metrics': {
                    'consistency_score': cognitive_metrics.consistency_score,
                    'sentiment_polarity_strength': cognitive_metrics.sentiment_polarity_strength,
                    'behavior_conversion_rate': cognitive_metrics.behavior_conversion_rate,
                    'cognitive_load_index': cognitive_metrics.cognitive_load_index,
                    'attitude_change_magnitude': cognitive_metrics.attitude_change_magnitude
                },
                'optimization': {
                    'strategy_adjustments': optimization_result.strategy_adjustments,
                    'parameter_tuning': optimization_result.parameter_tuning,
                    'prompt_optimization': optimization_result.prompt_optimization,
                    'expected_improvement': optimization_result.expected_improvement,
                    'confidence_score': optimization_result.confidence_score
                },
                'overall_assessment': self._calculate_overall_assessment(
                    effect_metrics, cognitive_metrics, optimization_result
                ),
                'recommendations': self._generate_comprehensive_recommendations(
                    current_metrics, optimization_result
                )
            }
            
            # 记录评估历史
            self.evaluation_history.append(evaluation_result)
            
            # 限制历史记录数量
            if len(self.evaluation_history) > 50:
                self.evaluation_history = self.evaluation_history[-50:]
            
            logger.info(f"EvalAgent评估完成 - 综合得分: {evaluation_result['overall_assessment']['composite_score']:.3f}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"EvalAgent分发性能评估失败: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _analyze_simulation_environment(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析模拟环境"""
        try:
            # 提取用户和内容数据
            user_interactions = simulation_data.get('user_interactions', [])
            unique_users = list(set(i.get('user_id') for i in user_interactions if i.get('user_id')))
            
            # 构建用户数据
            user_data = [{'user_id': uid, 'interests': ['general']} for uid in unique_users[:10]]
            
            # 构建内容数据
            content_data = [
                {
                    'content_id': f'content_{i}',
                    'text': f'模拟内容 {i}',
                    'topics': ['general'],
                    'complexity_score': 0.5
                }
                for i in range(3)
            ]
            
            # 初始化模拟环境
            env_state = await self.simulation_environment.initialize_simulation_environment(
                user_data, content_data
            )
            
            return {
                'environment_initialized': True,
                'user_count': len(user_data),
                'content_count': len(content_data),
                'environment_quality': env_state.get('environment_metrics', {}).get('simulation_quality_score', 0.8),
                'network_density': env_state.get('environment_metrics', {}).get('network_density', 0.3)
            }
            
        except Exception as e:
            logger.error(f"分析模拟环境失败: {e}")
            return {'error': str(e)}
    
    def _expectation_to_dict(self) -> Dict[str, float]:
        """将期望对象转换为字典"""
        return {
            'target_ctr': self.distribution_expectation.target_ctr,
            'target_engagement': self.distribution_expectation.target_engagement,
            'target_consistency': self.distribution_expectation.target_consistency,
            'target_conversion': self.distribution_expectation.target_conversion
        }
    
    def _calculate_overall_assessment(self, effect_metrics, cognitive_metrics, optimization_result) -> Dict[str, Any]:
        """计算综合评估"""
        try:
            # 效果得分 (0-1)
            effect_score = (
                effect_metrics.ctr * 0.25 +
                effect_metrics.deep_engagement_index * 0.25 +
                effect_metrics.propagation_influence * 0.25 +
                min(effect_metrics.reach_amplification / 5.0, 1.0) * 0.25
            )
            
            # 认知得分 (0-1)
            cognitive_score = (
                cognitive_metrics.consistency_score * 0.4 +
                cognitive_metrics.sentiment_polarity_strength * 0.3 +
                cognitive_metrics.behavior_conversion_rate * 0.3
            )
            
            # 优化潜力得分 (0-1)
            optimization_score = (
                optimization_result.expected_improvement * 0.6 +
                optimization_result.confidence_score * 0.4
            )
            
            # 综合得分
            composite_score = (effect_score * 0.4 + cognitive_score * 0.4 + optimization_score * 0.2)
            
            # 评估等级
            if composite_score >= 0.8:
                grade = 'A'
                assessment = '优秀'
            elif composite_score >= 0.6:
                grade = 'B'
                assessment = '良好'
            elif composite_score >= 0.4:
                grade = 'C'
                assessment = '一般'
            else:
                grade = 'D'
                assessment = '需改进'
            
            return {
                'composite_score': composite_score,
                'effect_score': effect_score,
                'cognitive_score': cognitive_score,
                'optimization_score': optimization_score,
                'grade': grade,
                'assessment': assessment,
                'score_breakdown': {
                    'effect_weight': 0.4,
                    'cognitive_weight': 0.4,
                    'optimization_weight': 0.2
                }
            }
            
        except Exception as e:
            logger.error(f"计算综合评估失败: {e}")
            return {'composite_score': 0.5, 'grade': 'C', 'assessment': '评估异常'}
    
    def _generate_comprehensive_recommendations(self, current_metrics: Dict[str, Any], 
                                              optimization_result) -> List[str]:
        """生成综合建议"""
        try:
            recommendations = []
            
            # 基于当前指标的建议
            if current_metrics.get('ctr', 0) < self.distribution_expectation.target_ctr:
                recommendations.append("提升内容吸引力，优化标题和开头，增加视觉元素")
            
            if current_metrics.get('cognitive_consistency', 0) < self.distribution_expectation.target_consistency:
                recommendations.append("加强内容与目标认知属性的匹配度，优化用户画像精准度")
            
            if current_metrics.get('behavior_conversion_rate', 0) < self.distribution_expectation.target_conversion:
                recommendations.append("增加明确的行动召唤元素，优化转化路径设计")
            
            # 基于优化结果的建议
            if optimization_result.expected_improvement > 0.1:
                recommendations.append(f"应用推荐的策略调整，预期可提升 {optimization_result.expected_improvement:.1%}")
            
            if optimization_result.confidence_score < 0.6:
                recommendations.append("当前优化建议置信度较低，建议收集更多数据后再次评估")
            
            # 基于策略调整的具体建议
            strategy = optimization_result.strategy_adjustments
            if strategy.get('content_strategy') == 'aggressive':
                recommendations.append("采用更激进的内容策略，增加争议性和话题性")
            elif strategy.get('user_targeting') == 'cognitive_aligned':
                recommendations.append("重点关注认知属性匹配的用户群体，提高精准度")
            
            return recommendations[:5]  # 限制建议数量
            
        except Exception as e:
            logger.error(f"生成综合建议失败: {e}")
            return ["建议收集更多数据进行深入分析"]
    
    def update_distribution_expectation(self, expectation: DistributionExpectation):
        """更新分发期望"""
        self.distribution_expectation = expectation
        logger.info("分发期望已更新")
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return self.evaluation_history
    
    def get_module_status(self) -> Dict[str, Any]:
        """获取各模块状态"""
        return {
            'data_collection': self.data_collection.get_collected_data_summary(),
            'simulation_environment': self.simulation_environment.get_simulation_state_summary(),
            'effect_analysis': self.effect_analysis.get_analysis_summary(),
            'cognitive_assessment': self.cognitive_assessment.get_assessment_summary(),
            'optimization_feedback': self.optimization_feedback.get_optimization_summary(),
            'evaluation_history_count': len(self.evaluation_history),
            'current_expectation': self._expectation_to_dict()
        }
    
    async def generate_comprehensive_report(self, evaluation_result: Dict[str, Any]) -> str:
        """生成综合评估报告"""
        try:
            report = f"""
EvalAgent 综合评估报告
=====================

评估ID: {evaluation_result.get('evaluation_id', 'unknown')}
评估时间: {evaluation_result.get('timestamp', 'unknown')}

一、数据收集分析
--------------
数据质量得分: {evaluation_result.get('data_collection', {}).get('data_quality_score', 0.0):.3f}
模拟数据条数: {evaluation_result.get('data_collection', {}).get('simulation_data_summary', {}).get('simulation_data_count', 0)}
实际数据条数: {evaluation_result.get('data_collection', {}).get('simulation_data_summary', {}).get('actual_data_count', 0)}

二、实际效果分析
--------------
点击率 (CTR): {evaluation_result.get('effect_metrics', {}).get('ctr', 0.0):.3f}
深度参与指数: {evaluation_result.get('effect_metrics', {}).get('deep_engagement_index', 0.0):.3f}
传播影响力: {evaluation_result.get('effect_metrics', {}).get('propagation_influence', 0.0):.3f}
触达放大系数: {evaluation_result.get('effect_metrics', {}).get('reach_amplification', 1.0):.2f}x

三、认知影响评估
--------------
认知一致性: {evaluation_result.get('cognitive_metrics', {}).get('consistency_score', 0.0):.3f}
情感极性强度: {evaluation_result.get('cognitive_metrics', {}).get('sentiment_polarity_strength', 0.0):.3f}
行为转化率: {evaluation_result.get('cognitive_metrics', {}).get('behavior_conversion_rate', 0.0):.3f}
认知负荷指数: {evaluation_result.get('cognitive_metrics', {}).get('cognitive_load_index', 0.0):.3f}

四、优化建议
----------
预期改进幅度: {evaluation_result.get('optimization', {}).get('expected_improvement', 0.0):.1%}
优化置信度: {evaluation_result.get('optimization', {}).get('confidence_score', 0.0):.3f}

策略调整:
- 内容策略: {evaluation_result.get('optimization', {}).get('strategy_adjustments', {}).get('content_strategy', 'unknown')}
- 用户定位: {evaluation_result.get('optimization', {}).get('strategy_adjustments', {}).get('user_targeting', 'unknown')}
- 时机策略: {evaluation_result.get('optimization', {}).get('strategy_adjustments', {}).get('timing_strategy', 'unknown')}

五、综合评估
----------
综合得分: {evaluation_result.get('overall_assessment', {}).get('composite_score', 0.0):.3f}
评估等级: {evaluation_result.get('overall_assessment', {}).get('grade', 'unknown')}
评估结论: {evaluation_result.get('overall_assessment', {}).get('assessment', 'unknown')}

六、行动建议
----------
"""
            
            recommendations = evaluation_result.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return "综合评估报告生成失败"

# 工厂函数
def create_eval_agent(config: Dict[str, Any] = None) -> EvalAgent:
    """创建EvalAgent实例"""
    return EvalAgent(config)
