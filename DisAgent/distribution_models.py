"""
分发策略数据结构定义
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import json


class TargetingCriteria(Enum):
    """目标定位标准"""
    DEMOGRAPHIC = "demographic"  # 人口统计学
    BEHAVIORAL = "behavioral"    # 行为特征
    PSYCHOGRAPHIC = "psychographic"  # 心理特征
    CONTEXTUAL = "contextual"    # 情境特征


@dataclass
class DistributionRule:
    """分发规则"""
    rule_id: str
    name: str
    description: str
    criteria_type: TargetingCriteria
    conditions: Dict[str, Any]  # 条件规则
    weight: float = 1.0  # 权重
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "criteria_type": self.criteria_type.value,
            "conditions": self.conditions,
            "weight": self.weight,
            "is_active": self.is_active
        }


@dataclass
class DistributionStrategy:
    """分发策略"""
    strategy_id: str
    name: str
    description: str
    rules: List[DistributionRule] = field(default_factory=list)
    target_metrics: Dict[str, float] = field(default_factory=dict)  # 目标指标
    created_at: str = ""
    version: str = "1.0"

    def add_rule(self, rule: DistributionRule):
        """添加规则"""
        self.rules.append(rule)

    def remove_rule(self, rule_id: str):
        """移除规则"""
        self.rules = [rule for rule in self.rules if rule.rule_id != rule_id]

    def get_active_rules(self) -> List[DistributionRule]:
        """获取激活的规则"""
        return [rule for rule in self.rules if rule.is_active]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "rules": [rule.to_dict() for rule in self.rules],
            "target_metrics": self.target_metrics,
            "created_at": self.created_at,
            "version": self.version
        }


@dataclass
class EvaluationMetrics:
    """评估指标"""
    engagement_rate: float = 0.0  # 参与率
    sentiment_shift: float = 0.0  # 情感变化
    opinion_diversity: float = 0.0  # 观点多样性
    reach: int = 0  # 触达人数
    conversion_rate: float = 0.0  # 转化率
    time_to_engagement: float = 0.0  # 参与时间
    virality_score: float = 0.0  # 病毒性评分

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engagement_rate": self.engagement_rate,
            "sentiment_shift": self.sentiment_shift,
            "opinion_diversity": self.opinion_diversity,
            "reach": self.reach,
            "conversion_rate": self.conversion_rate,
            "time_to_engagement": self.time_to_engagement,
            "virality_score": self.virality_score
        }

    def calculate_overall_score(self, weights: Dict[str, float] = None) -> float:
        """计算综合评分"""
        if weights is None:
            weights = {
                "engagement_rate": 0.2,
                "sentiment_shift": 0.15,
                "opinion_diversity": 0.15,
                "reach": 0.15,
                "conversion_rate": 0.15,
                "time_to_engagement": 0.1,
                "virality_score": 0.1
            }

        score = 0.0
        metrics_dict = self.to_dict()

        for metric, value in metrics_dict.items():
            if metric in weights:
                score += value * weights[metric]

        return score


@dataclass
class DistributionResult:
    """分发结果"""
    distribution_id: str
    strategy_id: str
    content_id: str
    target_users: List[str]  # 目标用户列表
    actual_recipients: List[str]  # 实际接收用户
    metrics: EvaluationMetrics
    timestamp: str
    duration: float  # 执行时长（秒）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distribution_id": self.distribution_id,
            "strategy_id": self.strategy_id,
            "content_id": self.content_id,
            "target_users": self.target_users,
            "actual_recipients": self.actual_recipients,
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp,
            "duration": self.duration
        }
