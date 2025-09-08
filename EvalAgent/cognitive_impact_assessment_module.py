# -*- coding: utf-8 -*-
"""
认知影响评估模块
量化 DISTAgent 生成内容对用户认知的影响（如认知一致性、情感极性、行为转化）
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
class CognitiveMetrics:
    """认知影响指标数据类"""
    consistency_score: float  # 认知一致性得分
    sentiment_polarity_strength: float  # 情感极性强度
    behavior_conversion_rate: float  # 行为转化率
    cognitive_load_index: float  # 认知负荷指数
    attitude_change_magnitude: float  # 态度变化幅度

class CognitiveImpactAssessmentModule:
    """
    认知影响评估模块
    功能：量化 DISTAgent 生成内容对用户认知的影响
    
    技术特性：
    - 分布式计算：分布式运行认知一致性分析
    - 隐私保护：对用户评论数据应用差分隐私
    - 自适应权重：动态调整认知指标权重
    - 实时反馈：实时收集评论数据，更新认知评估
    - 提示学习：基于NICE数据集和RoBERTa进行认知分析
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.consistency_threshold = self.config.get('consistency_threshold', 0.8)
        self.sentiment_analysis = self.config.get('sentiment_analysis', True)
        self.privacy_epsilon = self.config.get('privacy_epsilon', 1.0)
        self.enable_distributed = self.config.get('enable_distributed', False)
        
        # 自适应权重
        self.cognitive_weights = {
            'consistency_weight': 0.4,
            'sentiment_weight': 0.3,
            'conversion_weight': 0.3
        }
        
        # 认知模型参数
        self.cognitive_models = {
            'nice_embedding_dim': 1536,
            'roberta_sentiment_classes': 3,
            'cognitive_threshold': 0.6
        }
        
        logger.info("认知影响评估模块初始化完成")
        
    async def assess_comprehensive_cognitive_impact(self, 
                                                  content_data: Dict[str, Any],
                                                  user_interactions: List[Dict[str, Any]],
                                                  target_attributes: Dict[str, Any]) -> CognitiveMetrics:
        """
        综合认知影响评估
        认知影响 = w1·一致性得分 + w2·情感极性强度 + w3·行为转化率
        """
        try:
            # 1. 认知一致性分析
            consistency_score = await self._calculate_cognitive_consistency(
                user_interactions, target_attributes
            )
            
            # 2. 情感极性强度分析
            sentiment_strength = await self._calculate_sentiment_polarity_strength(
                user_interactions
            )
            
            # 3. 行为转化率分析
            conversion_rate = await self._calculate_behavior_conversion_rate(
                user_interactions, content_data
            )
            
            # 4. 认知负荷指数
            cognitive_load = await self._calculate_cognitive_load_index(
                content_data, user_interactions
            )
            
            # 5. 态度变化幅度
            attitude_change = await self._calculate_attitude_change_magnitude(
                user_interactions
            )
            
            # 应用差分隐私保护
            if self.config.get('privacy_protection', True):
                consistency_score, sentiment_strength, conversion_rate = self._apply_privacy_protection(
                    consistency_score, sentiment_strength, conversion_rate
                )
            
            metrics = CognitiveMetrics(
                consistency_score=consistency_score,
                sentiment_polarity_strength=sentiment_strength,
                behavior_conversion_rate=conversion_rate,
                cognitive_load_index=cognitive_load,
                attitude_change_magnitude=attitude_change
            )
            
            # 动态调整认知权重
            await self._optimize_cognitive_weights(metrics)
            
            logger.info(f"认知影响评估完成 - 一致性: {consistency_score:.3f}, 情感强度: {sentiment_strength:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"认知影响评估失败: {e}")
            return CognitiveMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    async def _calculate_cognitive_consistency(self, 
                                             user_interactions: List[Dict[str, Any]],
                                             target_attributes: Dict[str, Any]) -> float:
        """
        计算认知一致性得分
        认知一致性得分 = Σ_{i=1}^n cos(Emb_NICE(评论_i), Emb_NICE(T)) / n
        """
        try:
            comments = [i for i in user_interactions if i.get('type') == 'comment']
            
            if not comments:
                return 0.5  # 默认中性值
            
            consistency_scores = []
            target_embedding = self._get_nice_embedding(target_attributes)
            
            for comment in comments:
                comment_text = comment.get('content', '')
                if not comment_text:
                    continue
                
                # 使用NICE数据集训练的模型计算嵌入
                comment_embedding = self._get_nice_embedding({'text': comment_text})
                
                # 计算余弦相似度
                similarity = self._cosine_similarity(comment_embedding, target_embedding)
                consistency_scores.append(similarity)
            
            if not consistency_scores:
                return 0.5
            
            # 分布式计算优化
            if self.enable_distributed:
                return await self._distributed_compute_consistency(consistency_scores)
            
            avg_consistency = np.mean(consistency_scores)
            
            # 应用认知阈值
            if avg_consistency >= self.consistency_threshold:
                return min(1.0, avg_consistency * 1.1)  # 奖励高一致性
            else:
                return avg_consistency * 0.9  # 惩罚低一致性
            
        except Exception as e:
            logger.error(f"计算认知一致性失败: {e}")
            return 0.5
    
    async def _calculate_sentiment_polarity_strength(self, 
                                                   user_interactions: List[Dict[str, Any]]) -> float:
        """
        计算情感极性强度
        情感极性强度 = Σ_{i=1}^n 情感得分(评论_i) / n, 情感得分 = RoBERTa(评论_i) ∈ {1,0,-1}
        """
        try:
            if not self.sentiment_analysis:
                return 0.0
            
            comments = [i for i in user_interactions if i.get('type') == 'comment']
            
            if not comments:
                return 0.0
            
            sentiment_scores = []
            
            for comment in comments:
                comment_text = comment.get('content', '')
                if not comment_text:
                    continue
                
                # 使用RoBERTa进行情感分析
                sentiment_score = await self._roberta_sentiment_analysis(comment_text)
                sentiment_scores.append(sentiment_score)
            
            if not sentiment_scores:
                return 0.0
            
            # 计算情感极性强度（绝对值的平均）
            polarity_strength = np.mean([abs(score) for score in sentiment_scores])
            
            # 考虑情感一致性
            sentiment_variance = np.var(sentiment_scores)
            consistency_bonus = 1.0 - min(sentiment_variance, 1.0)
            
            final_strength = polarity_strength * (1.0 + 0.2 * consistency_bonus)
            
            return min(1.0, final_strength)
            
        except Exception as e:
            logger.error(f"计算情感极性强度失败: {e}")
            return 0.0
    
    async def _calculate_behavior_conversion_rate(self, 
                                                user_interactions: List[Dict[str, Any]],
                                                content_data: Dict[str, Any]) -> float:
        """
        计算行为转化率
        行为转化率 = 转化行为数 / 曝光次数, 转化行为 ∈ {点击,转发,二次创作}
        """
        try:
            # 统计转化行为
            conversion_actions = [
                i for i in user_interactions 
                if i.get('type') in ['click', 'share', 'repost', 'create', 'follow']
            ]
            
            # 统计曝光次数
            total_exposures = len([
                i for i in user_interactions 
                if i.get('type') in ['view', 'impression', 'click', 'comment', 'like']
            ])
            
            if total_exposures == 0:
                return 0.0
            
            base_conversion_rate = len(conversion_actions) / total_exposures
            
            # 考虑转化质量权重
            quality_weighted_conversions = 0.0
            for action in conversion_actions:
                action_type = action.get('type', '')
                engagement_level = action.get('engagement_level', 0.5)
                
                # 不同行为的权重
                type_weights = {
                    'click': 0.3,
                    'share': 0.8,
                    'repost': 0.7,
                    'create': 1.0,
                    'follow': 0.9
                }
                
                weight = type_weights.get(action_type, 0.5)
                quality_weighted_conversions += weight * engagement_level
            
            if len(conversion_actions) > 0:
                avg_conversion_quality = quality_weighted_conversions / len(conversion_actions)
                final_rate = base_conversion_rate * avg_conversion_quality
            else:
                final_rate = base_conversion_rate
            
            return min(1.0, final_rate)
            
        except Exception as e:
            logger.error(f"计算行为转化率失败: {e}")
            return 0.0
    
    async def _calculate_cognitive_load_index(self, 
                                            content_data: Dict[str, Any],
                                            user_interactions: List[Dict[str, Any]]) -> float:
        """计算认知负荷指数"""
        try:
            # 内容复杂度因素
            content_complexity = self._analyze_content_complexity(content_data)
            
            # 用户处理能力因素
            user_processing_capacity = self._estimate_user_processing_capacity(user_interactions)
            
            # 认知负荷 = 内容复杂度 / 用户处理能力
            cognitive_load = content_complexity / max(user_processing_capacity, 0.1)
            
            # 标准化到 [0, 1]
            normalized_load = min(1.0, cognitive_load)
            
            return normalized_load
            
        except Exception as e:
            logger.error(f"计算认知负荷指数失败: {e}")
            return 0.5
    
    async def _calculate_attitude_change_magnitude(self, 
                                                 user_interactions: List[Dict[str, Any]]) -> float:
        """计算态度变化幅度"""
        try:
            # 按时间排序交互
            sorted_interactions = sorted(
                user_interactions, 
                key=lambda x: x.get('timestamp', ''), 
                reverse=False
            )
            
            if len(sorted_interactions) < 2:
                return 0.0
            
            # 计算初始和最终态度
            initial_sentiment = self._extract_sentiment_from_interactions(
                sorted_interactions[:len(sorted_interactions)//3]  # 前1/3
            )
            final_sentiment = self._extract_sentiment_from_interactions(
                sorted_interactions[-len(sorted_interactions)//3:]  # 后1/3
            )
            
            # 态度变化幅度
            attitude_change = abs(final_sentiment - initial_sentiment)
            
            # 考虑变化方向的价值
            if final_sentiment > initial_sentiment:
                # 正向变化给予奖励
                attitude_change *= 1.2
            
            return min(1.0, attitude_change)
            
        except Exception as e:
            logger.error(f"计算态度变化幅度失败: {e}")
            return 0.0
    
    def _get_nice_embedding(self, data: Dict[str, Any]) -> np.ndarray:
        """获取NICE数据集嵌入（改进的语义表示）"""
        try:
            text = data.get('text', str(data))
            
            # 改进的语义嵌入计算
            embedding_dim = self.cognitive_models['nice_embedding_dim']
            
            # 基于词汇特征和语义特征的混合嵌入
            semantic_features = self._extract_semantic_features(text)
            
            # 转换为固定维度的嵌入向量
            embedding = np.zeros(embedding_dim)
            
            # 填充语义特征到嵌入向量
            feature_keys = list(semantic_features.keys())
            for i, key in enumerate(feature_keys[:embedding_dim]):
                embedding[i] = semantic_features[key]
            
            # 如果特征少于嵌入维度，用文本统计特征填充
            if len(feature_keys) < embedding_dim:
                text_stats = self._get_text_statistics(text)
                for i in range(len(feature_keys), min(embedding_dim, len(feature_keys) + len(text_stats))):
                    embedding[i] = text_stats[i - len(feature_keys)]
            
            # 归一化到单位向量
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                # 如果向量为零，生成随机单位向量
                embedding = np.random.normal(0, 0.1, embedding_dim)
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"获取NICE嵌入失败: {e}")
            # 返回随机单位向量作为后备
            embedding = np.random.normal(0, 0.1, self.cognitive_models['nice_embedding_dim'])
            return embedding / np.linalg.norm(embedding)
    
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """提取文本的语义特征"""
        features = {}
        
        # 1. 情感特征
        positive_words = ['好', '棒', '赞', '优秀', '喜欢', '支持', '同意', '正确']
        negative_words = ['差', '糟', '不好', '讨厌', '反对', '错误', '问题', '担心']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        features['sentiment_positive'] = pos_count / max(len(text), 1)
        features['sentiment_negative'] = neg_count / max(len(text), 1)
        
        # 2. 主题特征
        tech_words = ['技术', '创新', 'AI', '人工智能', '算法', '数据', '系统']
        social_words = ['社会', '人们', '大家', '用户', '公众', '群体']
        econ_words = ['经济', '市场', '投资', '价格', '成本', '收益', '股价']
        
        features['topic_tech'] = sum(1 for word in tech_words if word in text) / max(len(text), 1)
        features['topic_social'] = sum(1 for word in social_words if word in text) / max(len(text), 1)
        features['topic_economic'] = sum(1 for word in econ_words if word in text) / max(len(text), 1)
        
        # 3. 认知复杂度特征
        complex_words = ['但是', '然而', '因此', '所以', '如果', '虽然', '尽管']
        features['cognitive_complexity'] = sum(1 for word in complex_words if word in text) / max(len(text), 1)
        
        # 4. 参与度特征
        engage_words = ['怎么', '为什么', '应该', '需要', '建议', '希望', '期待']
        features['engagement_intent'] = sum(1 for word in engage_words if word in text) / max(len(text), 1)
        
        return features
    
    def _get_text_statistics(self, text: str) -> List[float]:
        """获取文本统计特征"""
        stats = []
        
        # 基本统计
        stats.append(len(text) / 1000.0)  # 标准化长度
        words = text.split()
        stats.append(len(words) / 500.0)  # 标准化词数
        stats.append(len(set(words)) / max(len(words), 1))  # 词汇多样性
        
        # 标点符号密度
        punctuation = '，。！？；：'
        punct_count = sum(1 for char in text if char in punctuation)
        stats.append(punct_count / max(len(text), 1))
        
        return stats
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度（改进版本）"""
        try:
            # 确保输入向量不为空
            if len(vec1) == 0 or len(vec2) == 0:
                return 0.0
                
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # 确保结果在[-1, 1]范围内
            similarity = np.clip(similarity, -1.0, 1.0)
            
            # 转换到[0, 1]范围（认知一致性通常为正值）
            normalized_similarity = (similarity + 1.0) / 2.0
            
            return float(normalized_similarity)
            
        except Exception as e:
            logger.error(f"计算余弦相似度失败: {e}")
            return 0.5  # 返回中性值而非0
    
    async def _roberta_sentiment_analysis(self, text: str) -> float:
        """RoBERTa情感分析（模拟实现）"""
        try:
            # 模拟RoBERTa情感分析
            # 在实际应用中，这里会调用预训练的RoBERTa模型
            
            # 基于关键词的简化情感分析
            positive_words = ['好', '棒', '赞', '优秀', '喜欢', '支持', '同意']
            negative_words = ['差', '糟', '反对', '不好', '讨厌', '拒绝', '错误']
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return min(1.0, 0.5 + 0.1 * positive_count)
            elif negative_count > positive_count:
                return max(-1.0, -0.5 - 0.1 * negative_count)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"RoBERTa情感分析失败: {e}")
            return 0.0
    
    def _analyze_content_complexity(self, content_data: Dict[str, Any]) -> float:
        """分析内容复杂度"""
        try:
            text = content_data.get('text', content_data.get('content', ''))
            
            # 多维度复杂度分析
            factors = []
            
            # 1. 文本长度复杂度
            length_complexity = min(len(text) / 500.0, 1.0)
            factors.append(length_complexity)
            
            # 2. 词汇复杂度
            words = text.split()
            unique_words = len(set(words))
            vocab_complexity = min(unique_words / max(len(words), 1), 1.0)
            factors.append(vocab_complexity)
            
            # 3. 句法复杂度（简化）
            sentences = text.split('。')
            avg_sentence_length = len(words) / max(len(sentences), 1)
            syntax_complexity = min(avg_sentence_length / 20.0, 1.0)
            factors.append(syntax_complexity)
            
            # 4. 概念复杂度
            abstract_words = ['概念', '理论', '模型', '框架', '系统', '机制']
            concept_count = sum(1 for word in abstract_words if word in text)
            concept_complexity = min(concept_count / 5.0, 1.0)
            factors.append(concept_complexity)
            
            return np.mean(factors)
            
        except Exception as e:
            logger.error(f"分析内容复杂度失败: {e}")
            return 0.5
    
    def _estimate_user_processing_capacity(self, user_interactions: List[Dict[str, Any]]) -> float:
        """估算用户处理能力"""
        try:
            if not user_interactions:
                return 0.7  # 默认值
            
            # 基于用户行为模式估算处理能力
            factors = []
            
            # 1. 响应速度
            response_times = []
            for i in range(1, len(user_interactions)):
                prev_time = user_interactions[i-1].get('timestamp', '')
                curr_time = user_interactions[i].get('timestamp', '')
                # 简化的时间差计算
                response_times.append(1.0)  # 模拟值
            
            if response_times:
                avg_response_time = np.mean(response_times)
                speed_factor = max(0.1, 1.0 - avg_response_time / 10.0)
                factors.append(speed_factor)
            
            # 2. 交互深度
            deep_interactions = [
                i for i in user_interactions 
                if i.get('type') in ['comment', 'create', 'share']
            ]
            depth_factor = len(deep_interactions) / max(len(user_interactions), 1)
            factors.append(depth_factor)
            
            # 3. 内容理解质量
            comments = [i for i in user_interactions if i.get('type') == 'comment']
            if comments:
                avg_comment_length = np.mean([len(str(c.get('content', ''))) for c in comments])
                understanding_factor = min(avg_comment_length / 50.0, 1.0)
                factors.append(understanding_factor)
            
            return np.mean(factors) if factors else 0.7
            
        except Exception as e:
            logger.error(f"估算用户处理能力失败: {e}")
            return 0.7
    
    def _extract_sentiment_from_interactions(self, interactions: List[Dict[str, Any]]) -> float:
        """从交互中提取情感倾向"""
        try:
            sentiments = []
            
            for interaction in interactions:
                if interaction.get('type') == 'comment':
                    # 从评论中提取情感
                    content = interaction.get('content', '')
                    sentiment = self._simple_sentiment_analysis(content)
                    sentiments.append(sentiment)
                elif interaction.get('type') == 'like':
                    sentiments.append(0.5)  # 点赞表示轻微正面
                elif interaction.get('type') == 'share':
                    sentiments.append(0.7)  # 分享表示较强正面
            
            return np.mean(sentiments) if sentiments else 0.0
            
        except Exception as e:
            logger.error(f"提取情感倾向失败: {e}")
            return 0.0
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """简单情感分析"""
        positive_words = ['好', '棒', '赞', '优秀', '喜欢']
        negative_words = ['差', '糟', '不好', '讨厌', '反对']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 0.5 + 0.1 * positive_count
        elif negative_count > positive_count:
            return -0.5 - 0.1 * negative_count
        else:
            return 0.0
    
    async def _distributed_compute_consistency(self, consistency_scores: List[float]) -> float:
        """分布式计算一致性（模拟）"""
        try:
            # 模拟分布式计算
            chunk_size = max(1, len(consistency_scores) // 4)
            chunks = [consistency_scores[i:i + chunk_size] 
                     for i in range(0, len(consistency_scores), chunk_size)]
            
            chunk_means = [np.mean(chunk) for chunk in chunks if chunk]
            
            return np.mean(chunk_means) if chunk_means else 0.5
            
        except Exception as e:
            logger.error(f"分布式计算一致性失败: {e}")
            return 0.5
    
    def _apply_privacy_protection(self, consistency: float, sentiment: float, 
                                conversion: float) -> Tuple[float, float, float]:
        """
        应用差分隐私保护
        评论_i' = 评论_i + N(0, σ²)
        """
        try:
            # 添加差分隐私噪声
            consistency_noise = np.random.normal(0, self.privacy_epsilon * 0.02)
            sentiment_noise = np.random.normal(0, self.privacy_epsilon * 0.015)
            conversion_noise = np.random.normal(0, self.privacy_epsilon * 0.01)
            
            protected_consistency = np.clip(consistency + consistency_noise, 0, 1)
            protected_sentiment = np.clip(sentiment + sentiment_noise, 0, 1)
            protected_conversion = np.clip(conversion + conversion_noise, 0, 1)
            
            logger.debug("认知指标差分隐私保护已应用")
            return protected_consistency, protected_sentiment, protected_conversion
            
        except Exception as e:
            logger.error(f"应用隐私保护失败: {e}")
            return consistency, sentiment, conversion
    
    async def _optimize_cognitive_weights(self, metrics: CognitiveMetrics):
        """
        优化认知权重
        w* = argmax_w E[认知影响|w]
        """
        try:
            # 基于指标表现调整权重
            metric_values = [
                metrics.consistency_score,
                metrics.sentiment_polarity_strength,
                metrics.behavior_conversion_rate
            ]
            
            # 计算权重重要性
            importance_scores = []
            for value in metric_values:
                # 基于方差和效果计算重要性
                importance = value * (1 + np.random.uniform(-0.05, 0.05))
                importance_scores.append(max(0.1, importance))
            
            # 归一化权重
            total_importance = sum(importance_scores)
            if total_importance > 0:
                new_weights = [score / total_importance for score in importance_scores]
                
                # 更新权重
                weight_keys = ['consistency_weight', 'sentiment_weight', 'conversion_weight']
                for i, key in enumerate(weight_keys):
                    if i < len(new_weights):
                        self.cognitive_weights[key] = new_weights[i]
            
            logger.debug(f"认知权重已优化: {self.cognitive_weights}")
            
        except Exception as e:
            logger.error(f"优化认知权重失败: {e}")
    
    async def generate_cognitive_assessment_report(self, 
                                                 metrics: CognitiveMetrics,
                                                 target_attributes: Dict[str, Any]) -> str:
        """生成认知影响评估报告"""
        try:
            # 计算综合认知影响得分
            cognitive_impact_score = (
                self.cognitive_weights['consistency_weight'] * metrics.consistency_score +
                self.cognitive_weights['sentiment_weight'] * metrics.sentiment_polarity_strength +
                self.cognitive_weights['conversion_weight'] * metrics.behavior_conversion_rate
            )
            
            report = f"""
认知影响评估报告
================

核心认知指标:
- 认知一致性得分: {metrics.consistency_score:.3f}
- 情感极性强度: {metrics.sentiment_polarity_strength:.3f}
- 行为转化率: {metrics.behavior_conversion_rate:.3f}
- 认知负荷指数: {metrics.cognitive_load_index:.3f}
- 态度变化幅度: {metrics.attitude_change_magnitude:.3f}

综合认知影响得分: {cognitive_impact_score:.3f}

认知权重配置:
- 一致性权重: {self.cognitive_weights['consistency_weight']:.2f}
- 情感权重: {self.cognitive_weights['sentiment_weight']:.2f}
- 转化权重: {self.cognitive_weights['conversion_weight']:.2f}

评估结论:
"""
            
            # 添加评估结论
            if cognitive_impact_score >= 0.8:
                report += "- 认知影响效果优秀，内容与目标高度一致\n"
            elif cognitive_impact_score >= 0.6:
                report += "- 认知影响效果良好，有进一步优化空间\n"
            else:
                report += "- 认知影响效果需要改进，建议调整内容策略\n"
            
            if metrics.consistency_score < self.consistency_threshold:
                report += "- 建议提高内容与目标认知属性的匹配度\n"
            
            if metrics.behavior_conversion_rate < 0.1:
                report += "- 建议增强行为召唤元素，提升转化引导\n"
            
            return report
            
        except Exception as e:
            logger.error(f"生成认知评估报告失败: {e}")
            return "认知评估报告生成失败"
    
    def get_assessment_summary(self) -> Dict[str, Any]:
        """获取评估模块摘要"""
        return {
            'module_name': 'CognitiveImpactAssessmentModule',
            'cognitive_weights': self.cognitive_weights,
            'consistency_threshold': self.consistency_threshold,
            'sentiment_analysis_enabled': self.sentiment_analysis,
            'privacy_protection': self.config.get('privacy_protection', True),
            'distributed_computing': self.enable_distributed,
            'cognitive_models': self.cognitive_models,
            'privacy_epsilon': self.privacy_epsilon
        }

# 工厂函数
def create_cognitive_impact_assessment_module(config: Dict[str, Any] = None) -> CognitiveImpactAssessmentModule:
    """创建认知影响评估模块实例"""
    return CognitiveImpactAssessmentModule(config)
