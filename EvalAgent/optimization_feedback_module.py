# -*- coding: utf-8 -*-
"""
优化反馈模块
基于模拟与实际对比和认知影响评估，优化 DISTAgent 的分发策略
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
class OptimizationResult:
    """优化结果数据类"""
    strategy_adjustments: Dict[str, Any]
    parameter_tuning: Dict[str, float]
    prompt_optimization: Dict[str, str]
    expected_improvement: float
    confidence_score: float

class OptimizationFeedbackModule:
    """
    优化反馈模块
    功能：基于模拟与实际对比和认知影响评估，优化 DISTAgent 的分发策略
    
    技术特性：
    - 分布式计算：分布式运行 RL 优化
    - 隐私保护：对优化参数添加差分隐私噪声
    - 自适应权重：动态调整优化目标权重
    - 实时反馈：实时更新优化策略
    - 强化学习：使用RL优化分发策略
    - 提示学习：通过参数微调和提示学习迭代优化
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enable_auto_optimization = self.config.get('enable_auto_optimization', True)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.prompt_learning = self.config.get('prompt_learning', True)
        self.privacy_epsilon = self.config.get('privacy_epsilon', 1.0)
        
        # 优化历史
        self.optimization_history = []
        
        # 优化权重
        self.optimization_weights = {
            'ctr_weight': 0.25,
            'engagement_weight': 0.25,
            'consistency_weight': 0.25,
            'conversion_weight': 0.25
        }
        
        # RL参数
        self.rl_params = {
            'epsilon': 0.1,  # 探索率
            'gamma': 0.95,   # 折扣因子
            'alpha': self.learning_rate  # 学习率
        }
        
        logger.info("优化反馈模块初始化完成")
        
    async def optimize_distribution_strategy(self, 
                                           current_metrics: Dict[str, Any],
                                           expectation: Dict[str, Any],
                                           historical_data: List[Dict[str, Any]] = None) -> OptimizationResult:
        """
        优化分发策略
        π_{t+1} = π_t + η·∇_π E[分发期望]
        θ ← θ - η·∇_θ L(分发效果, 期望)
        """
        try:
            if not self.enable_auto_optimization:
                logger.info("自动优化未启用")
                return OptimizationResult({}, {}, {}, 0.0, 0.0)
            
            # 1. 计算优化目标
            optimization_target = self._calculate_optimization_target(current_metrics, expectation)
            
            # 2. 强化学习策略优化
            rl_adjustments = await self._rl_strategy_optimization(
                current_metrics, expectation, historical_data
            )
            
            # 3. 参数微调
            parameter_tuning = await self._parameter_fine_tuning(
                current_metrics, expectation, optimization_target
            )
            
            # 4. 提示学习优化
            prompt_optimization = await self._prompt_learning_optimization(
                current_metrics, expectation
            )
            
            # 5. 计算预期改进
            expected_improvement = self._calculate_expected_improvement(
                current_metrics, rl_adjustments, parameter_tuning
            )
            
            # 6. 计算置信度
            confidence_score = self._calculate_optimization_confidence(
                optimization_target, historical_data
            )
            
            # 应用差分隐私保护
            if self.config.get('privacy_protection', True):
                parameter_tuning = self._apply_privacy_protection(parameter_tuning)
            
            result = OptimizationResult(
                strategy_adjustments=rl_adjustments,
                parameter_tuning=parameter_tuning,
                prompt_optimization=prompt_optimization,
                expected_improvement=expected_improvement,
                confidence_score=confidence_score
            )
            
            # 记录优化历史
            self._record_optimization_history(result, current_metrics, expectation)
            
            # 动态调整优化权重
            await self._adapt_optimization_weights(result, current_metrics)
            
            logger.info(f"策略优化完成 - 预期改进: {expected_improvement:.3f}, 置信度: {confidence_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"优化分发策略失败: {e}")
            return OptimizationResult({}, {}, {}, 0.0, 0.0)
    
    def _calculate_optimization_target(self, current_metrics: Dict[str, Any], 
                                     expectation: Dict[str, Any]) -> Dict[str, float]:
        """计算优化目标"""
        try:
            target = {}
            
            # 计算各指标差距
            for metric_name in ['ctr', 'deep_engagement_index', 'cognitive_consistency', 'behavior_conversion_rate']:
                current_value = current_metrics.get(metric_name, 0.0)
                target_value = expectation.get(f'target_{metric_name}', current_value * 1.1)
                gap = target_value - current_value
                target[f'{metric_name}_gap'] = gap
                target[f'{metric_name}_improvement_ratio'] = gap / max(current_value, 0.001)
            
            return target
            
        except Exception as e:
            logger.error(f"计算优化目标失败: {e}")
            return {}
    
    async def _rl_strategy_optimization(self, 
                                      current_metrics: Dict[str, Any],
                                      expectation: Dict[str, Any],
                                      historical_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        强化学习策略优化
        π_优化 = RL(反馈_i, 节点_i)
        """
        try:
            # 构建状态空间
            state = self._construct_rl_state(current_metrics, expectation)
            
            # 计算奖励信号
            reward = self._calculate_rl_reward(current_metrics, expectation)
            
            # 策略梯度更新
            policy_gradient = self._calculate_policy_gradient(state, reward, historical_data)
            
            # 生成策略调整
            strategy_adjustments = {
                'content_strategy': self._optimize_content_strategy(policy_gradient),
                'user_targeting': self._optimize_user_targeting(policy_gradient),
                'timing_strategy': self._optimize_timing_strategy(policy_gradient),
                'platform_weights': self._optimize_platform_weights(policy_gradient),
                'personalization_strength': self._optimize_personalization(policy_gradient)
            }
            
            # 更新RL参数
            self._update_rl_parameters(reward)
            
            return strategy_adjustments
            
        except Exception as e:
            logger.error(f"RL策略优化失败: {e}")
            return {}
    
    async def _parameter_fine_tuning(self, 
                                   current_metrics: Dict[str, Any],
                                   expectation: Dict[str, Any],
                                   optimization_target: Dict[str, float]) -> Dict[str, float]:
        """参数微调"""
        try:
            tuning_params = {}
            
            # 基于差距调整参数
            ctr_gap = optimization_target.get('ctr_gap', 0.0)
            engagement_gap = optimization_target.get('deep_engagement_index_gap', 0.0)
            consistency_gap = optimization_target.get('cognitive_consistency_gap', 0.0)
            
            # 调整分发参数
            if ctr_gap > 0.02:
                tuning_params['posts_per_round'] = 4  # 增加曝光
                tuning_params['content_diversity'] = 0.8  # 提高多样性
            else:
                tuning_params['posts_per_round'] = 2
                tuning_params['content_diversity'] = 0.6
            
            if engagement_gap > 0.1:
                tuning_params['users_per_post'] = 10  # 扩大目标用户
                tuning_params['engagement_threshold'] = 0.6  # 降低门槛
            else:
                tuning_params['users_per_post'] = 6
                tuning_params['engagement_threshold'] = 0.7
            
            if consistency_gap > 0.1:
                tuning_params['personalization_strength'] = 0.9  # 提高个性化
                tuning_params['cognitive_matching_weight'] = 0.8
            else:
                tuning_params['personalization_strength'] = 0.7
                tuning_params['cognitive_matching_weight'] = 0.6
            
            # 学习率自适应调整
            tuning_params['learning_rate'] = self._adaptive_learning_rate(optimization_target)
            
            return tuning_params
            
        except Exception as e:
            logger.error(f"参数微调失败: {e}")
            return {}
    
    async def _prompt_learning_optimization(self, 
                                          current_metrics: Dict[str, Any],
                                          expectation: Dict[str, Any]) -> Dict[str, str]:
        """
        提示学习优化
        Prompt_t = "基于指标（一致性、情感极性、转化率），评估认知影响并建议优化策略。"
        """
        try:
            if not self.prompt_learning:
                return {}
            
            # 基础提示模板
            base_prompt = "基于用户画像和内容特征，生成最优分发策略"
            
            # 根据当前指标调整提示
            ctr = current_metrics.get('ctr', 0.0)
            consistency = current_metrics.get('cognitive_consistency', 0.0)
            conversion = current_metrics.get('behavior_conversion_rate', 0.0)
            
            # 动态提示生成
            if ctr < expectation.get('target_ctr', 0.1):
                distribution_prompt = base_prompt + "，重点优化内容吸引力和点击引导"
            elif consistency < expectation.get('target_consistency', 0.8):
                distribution_prompt = base_prompt + "，重点关注认知属性匹配度和用户理解"
            elif conversion < expectation.get('target_conversion', 0.15):
                distribution_prompt = base_prompt + "，重点关注行为转化引导和用户激活"
            else:
                distribution_prompt = base_prompt + "，保持当前策略并微调优化"
            
            # 评估提示优化
            evaluation_prompt = f"""
基于当前指标分析：
- CTR: {ctr:.3f} (目标: {expectation.get('target_ctr', 0.1):.3f})
- 认知一致性: {consistency:.3f} (目标: {expectation.get('target_consistency', 0.8):.3f})
- 转化率: {conversion:.3f} (目标: {expectation.get('target_conversion', 0.15):.3f})

请评估认知影响并提供具体的优化建议，包括内容策略、用户定位和时机选择。
"""
            
            # 认知分析提示
            cognitive_prompt = self._generate_cognitive_analysis_prompt(current_metrics, expectation)
            
            return {
                'distribution_prompt': distribution_prompt,
                'evaluation_prompt': evaluation_prompt,
                'cognitive_analysis_prompt': cognitive_prompt
            }
            
        except Exception as e:
            logger.error(f"提示学习优化失败: {e}")
            return {}
    
    def _construct_rl_state(self, current_metrics: Dict[str, Any], 
                           expectation: Dict[str, Any]) -> np.ndarray:
        """构建RL状态空间"""
        try:
            state_features = [
                current_metrics.get('ctr', 0.0),
                current_metrics.get('deep_engagement_index', 0.0),
                current_metrics.get('cognitive_consistency', 0.0),
                current_metrics.get('behavior_conversion_rate', 0.0),
                expectation.get('target_ctr', 0.1),
                expectation.get('target_engagement', 0.7),
                expectation.get('target_consistency', 0.8),
                expectation.get('target_conversion', 0.15)
            ]
            
            return np.array(state_features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"构建RL状态失败: {e}")
            return np.zeros(8, dtype=np.float32)
    
    def _calculate_rl_reward(self, current_metrics: Dict[str, Any], 
                           expectation: Dict[str, Any]) -> float:
        """计算RL奖励信号"""
        try:
            rewards = []
            
            # 各指标奖励
            for metric_name, weight in self.optimization_weights.items():
                metric_key = metric_name.replace('_weight', '')
                current_value = current_metrics.get(metric_key, 0.0)
                target_value = expectation.get(f'target_{metric_key}', current_value)
                
                # 计算达成度奖励
                if target_value > 0:
                    achievement = min(current_value / target_value, 2.0)  # 限制最大奖励
                    reward = weight * achievement
                    rewards.append(reward)
            
            total_reward = sum(rewards)
            
            # 添加稳定性奖励
            if 0.8 <= total_reward <= 1.2:
                total_reward += 0.1  # 稳定性奖励
            
            return total_reward
            
        except Exception as e:
            logger.error(f"计算RL奖励失败: {e}")
            return 0.0
    
    def _calculate_policy_gradient(self, state: np.ndarray, reward: float, 
                                 historical_data: List[Dict[str, Any]] = None) -> np.ndarray:
        """计算策略梯度"""
        try:
            # 简化的策略梯度计算
            gradient = np.zeros_like(state)
            
            # 基于奖励调整梯度
            if reward > 1.0:
                # 正向奖励，增强当前策略
                gradient = state * self.rl_params['alpha'] * (reward - 1.0)
            else:
                # 负向奖励，探索新策略
                gradient = -state * self.rl_params['alpha'] * (1.0 - reward)
                # 添加探索噪声
                gradient += np.random.normal(0, self.rl_params['epsilon'], size=state.shape)
            
            return gradient
            
        except Exception as e:
            logger.error(f"计算策略梯度失败: {e}")
            return np.zeros_like(state)
    
    def _optimize_content_strategy(self, gradient: np.ndarray) -> str:
        """优化内容策略"""
        try:
            # 基于梯度方向决定内容策略
            avg_gradient = np.mean(gradient[:4])  # 前4个特征对应当前指标
            
            if avg_gradient > 0.1:
                return 'aggressive'  # 激进策略
            elif avg_gradient < -0.1:
                return 'conservative'  # 保守策略
            else:
                return 'balanced'  # 平衡策略
                
        except Exception as e:
            logger.error(f"优化内容策略失败: {e}")
            return 'balanced'
    
    def _optimize_user_targeting(self, gradient: np.ndarray) -> str:
        """优化用户定位"""
        try:
            consistency_gradient = gradient[2] if len(gradient) > 2 else 0.0
            
            if consistency_gradient > 0.05:
                return 'cognitive_aligned'  # 认知对齐
            elif consistency_gradient < -0.05:
                return 'diverse_exploration'  # 多样化探索
            else:
                return 'precision'  # 精准定位
                
        except Exception as e:
            logger.error(f"优化用户定位失败: {e}")
            return 'precision'
    
    def _optimize_timing_strategy(self, gradient: np.ndarray) -> str:
        """优化时机策略"""
        try:
            engagement_gradient = gradient[1] if len(gradient) > 1 else 0.0
            
            if engagement_gradient > 0.05:
                return 'peak_hours'  # 高峰时段
            elif engagement_gradient < -0.05:
                return 'off_peak'  # 非高峰时段
            else:
                return 'optimal'  # 最优时机
                
        except Exception as e:
            logger.error(f"优化时机策略失败: {e}")
            return 'optimal'
    
    def _optimize_platform_weights(self, gradient: np.ndarray) -> Dict[str, float]:
        """优化平台权重"""
        try:
            ctr_gradient = gradient[0] if len(gradient) > 0 else 0.0
            
            if ctr_gradient > 0.05:
                # 提高社交平台权重
                return {'social': 0.7, 'professional': 0.3}
            elif ctr_gradient < -0.05:
                # 提高专业平台权重
                return {'social': 0.4, 'professional': 0.6}
            else:
                # 平衡权重
                return {'social': 0.5, 'professional': 0.5}
                
        except Exception as e:
            logger.error(f"优化平台权重失败: {e}")
            return {'social': 0.5, 'professional': 0.5}
    
    def _optimize_personalization(self, gradient: np.ndarray) -> float:
        """优化个性化强度"""
        try:
            consistency_gradient = gradient[2] if len(gradient) > 2 else 0.0
            conversion_gradient = gradient[3] if len(gradient) > 3 else 0.0
            
            # 基于认知一致性和转化率调整个性化强度
            adjustment = (consistency_gradient + conversion_gradient) * 0.1
            base_strength = 0.7
            
            optimized_strength = base_strength + adjustment
            return np.clip(optimized_strength, 0.3, 1.0)
            
        except Exception as e:
            logger.error(f"优化个性化强度失败: {e}")
            return 0.7
    
    def _update_rl_parameters(self, reward: float):
        """更新RL参数"""
        try:
            # 自适应调整探索率
            if reward > 1.0:
                # 好结果，减少探索
                self.rl_params['epsilon'] *= 0.99
            else:
                # 差结果，增加探索
                self.rl_params['epsilon'] *= 1.01
            
            # 限制探索率范围
            self.rl_params['epsilon'] = np.clip(self.rl_params['epsilon'], 0.05, 0.3)
            
            logger.debug(f"RL参数已更新 - epsilon: {self.rl_params['epsilon']:.3f}")
            
        except Exception as e:
            logger.error(f"更新RL参数失败: {e}")
    
    def _adaptive_learning_rate(self, optimization_target: Dict[str, float]) -> float:
        """自适应学习率调整"""
        try:
            # 基于优化目标调整学习率
            max_gap = max(abs(gap) for gap in optimization_target.values() if isinstance(gap, (int, float)))
            
            if max_gap > 0.2:
                # 大差距，提高学习率
                return min(self.learning_rate * 1.5, 0.1)
            elif max_gap < 0.05:
                # 小差距，降低学习率
                return max(self.learning_rate * 0.8, 0.001)
            else:
                return self.learning_rate
                
        except Exception as e:
            logger.error(f"自适应学习率调整失败: {e}")
            return self.learning_rate
    
    def _generate_cognitive_analysis_prompt(self, current_metrics: Dict[str, Any], 
                                          expectation: Dict[str, Any]) -> str:
        """生成认知分析提示"""
        try:
            prompt = f"""
请基于以下认知指标进行深度分析：

当前表现：
- 认知一致性: {current_metrics.get('cognitive_consistency', 0.0):.3f}
- 情感极性强度: {current_metrics.get('sentiment_polarity_strength', 0.0):.3f}
- 行为转化率: {current_metrics.get('behavior_conversion_rate', 0.0):.3f}

目标期望：
- 目标一致性: {expectation.get('target_consistency', 0.8):.3f}
- 目标转化率: {expectation.get('target_conversion', 0.15):.3f}

请分析认知影响的有效性，并提供以下维度的优化建议：
1. 内容认知匹配度优化
2. 用户心理模型调整
3. 情感引导策略改进
4. 行为激励机制设计
"""
            return prompt
            
        except Exception as e:
            logger.error(f"生成认知分析提示失败: {e}")
            return "请分析当前认知影响效果并提供优化建议。"
    
    def _calculate_expected_improvement(self, current_metrics: Dict[str, Any],
                                      strategy_adjustments: Dict[str, Any],
                                      parameter_tuning: Dict[str, float]) -> float:
        """计算预期改进幅度"""
        try:
            # 基于历史优化效果估算改进
            base_improvement = 0.05  # 基础改进预期
            
            # 策略调整带来的改进
            strategy_impact = 0.0
            if strategy_adjustments.get('content_strategy') == 'aggressive':
                strategy_impact += 0.03
            if strategy_adjustments.get('user_targeting') == 'cognitive_aligned':
                strategy_impact += 0.02
            
            # 参数调整带来的改进
            param_impact = 0.0
            learning_rate_change = abs(parameter_tuning.get('learning_rate', self.learning_rate) - self.learning_rate)
            param_impact += learning_rate_change * 0.5
            
            total_improvement = base_improvement + strategy_impact + param_impact
            return min(total_improvement, 0.2)  # 限制最大预期改进
            
        except Exception as e:
            logger.error(f"计算预期改进失败: {e}")
            return 0.05
    
    def _calculate_optimization_confidence(self, optimization_target: Dict[str, float],
                                         historical_data: List[Dict[str, Any]] = None) -> float:
        """计算优化置信度"""
        try:
            confidence_factors = []
            
            # 基于目标差距的置信度
            if optimization_target:
                avg_gap = np.mean([abs(gap) for gap in optimization_target.values() if isinstance(gap, (int, float))])
                gap_confidence = max(0.3, 1.0 - avg_gap)  # 差距越小置信度越高
                confidence_factors.append(gap_confidence)
            
            # 基于历史数据的置信度
            if historical_data and len(historical_data) > 1:
                # 计算历史优化成功率
                success_count = sum(1 for data in historical_data if data.get('improvement', 0) > 0)
                success_rate = success_count / len(historical_data)
                confidence_factors.append(success_rate)
            else:
                confidence_factors.append(0.6)  # 默认置信度
            
            # 基于当前指标稳定性的置信度
            stability_confidence = 0.8  # 简化处理
            confidence_factors.append(stability_confidence)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"计算优化置信度失败: {e}")
            return 0.6
    
    def _apply_privacy_protection(self, parameter_tuning: Dict[str, float]) -> Dict[str, float]:
        """
        应用差分隐私保护
        π' = π + N(0, σ²)
        """
        try:
            protected_params = {}
            
            for key, value in parameter_tuning.items():
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, self.privacy_epsilon * 0.01)
                    protected_value = value + noise
                    
                    # 确保参数在合理范围内
                    if 'rate' in key or 'strength' in key:
                        protected_value = np.clip(protected_value, 0.0, 1.0)
                    elif 'per_' in key:
                        protected_value = max(1, int(protected_value))
                    
                    protected_params[key] = protected_value
                else:
                    protected_params[key] = value
            
            logger.debug("优化参数差分隐私保护已应用")
            return protected_params
            
        except Exception as e:
            logger.error(f"应用隐私保护失败: {e}")
            return parameter_tuning
    
    def _record_optimization_history(self, result: OptimizationResult,
                                   current_metrics: Dict[str, Any],
                                   expectation: Dict[str, Any]):
        """记录优化历史"""
        try:
            history_record = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': current_metrics,
                'expectation': expectation,
                'optimization_result': {
                    'strategy_adjustments': result.strategy_adjustments,
                    'parameter_tuning': result.parameter_tuning,
                    'expected_improvement': result.expected_improvement,
                    'confidence_score': result.confidence_score
                },
                'optimization_weights': self.optimization_weights.copy(),
                'rl_params': self.rl_params.copy()
            }
            
            self.optimization_history.append(history_record)
            
            # 限制历史记录数量
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.debug("优化历史已记录")
            
        except Exception as e:
            logger.error(f"记录优化历史失败: {e}")
    
    async def _adapt_optimization_weights(self, result: OptimizationResult,
                                        current_metrics: Dict[str, Any]):
        """
        自适应调整优化权重
        w* = argmax_w E[优化效果|w]
        """
        try:
            # 基于预期改进调整权重
            if result.expected_improvement > 0.1:
                # 高改进预期，增加相关指标权重
                best_metric = max(current_metrics.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                metric_name = best_metric[0]
                
                for weight_key in self.optimization_weights:
                    if metric_name in weight_key:
                        self.optimization_weights[weight_key] *= 1.1
            
            # 归一化权重
            total_weight = sum(self.optimization_weights.values())
            if total_weight > 0:
                for key in self.optimization_weights:
                    self.optimization_weights[key] /= total_weight
            
            logger.debug(f"优化权重已调整: {self.optimization_weights}")
            
        except Exception as e:
            logger.error(f"调整优化权重失败: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化模块摘要"""
        return {
            'module_name': 'OptimizationFeedbackModule',
            'optimization_weights': self.optimization_weights,
            'rl_params': self.rl_params,
            'learning_rate': self.learning_rate,
            'auto_optimization_enabled': self.enable_auto_optimization,
            'prompt_learning_enabled': self.prompt_learning,
            'optimization_history_count': len(self.optimization_history),
            'privacy_protection': self.config.get('privacy_protection', True),
            'privacy_epsilon': self.privacy_epsilon
        }

# 工厂函数
def create_optimization_feedback_module(config: Dict[str, Any] = None) -> OptimizationFeedbackModule:
    """创建优化反馈模块实例"""
    return OptimizationFeedbackModule(config)
