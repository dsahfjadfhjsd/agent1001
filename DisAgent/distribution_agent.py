"""
分发智能体核心实现
"""
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from openai import OpenAI

from .distribution_models import (
    DistributionStrategy, DistributionRule, EvaluationMetrics,
    DistributionResult, TargetingCriteria
)
from UserAgent.user_profile import UserProfile
from SimulateEnv.environment_manager import EnvironmentManager
from SimulateEnv.environment_models import Post
from Config.settings import config


class DistributionAgent:
    """分发智能体"""

    def __init__(self, strategy: DistributionStrategy = None):
        self.strategy = strategy
        self.client = OpenAI(
            api_key=config.model.api_key,
            base_url=config.model.base_url
        )
        self.performance_history: List[DistributionResult] = []

    def _build_targeting_prompt(self, content: Dict[str, Any], available_users: List[UserProfile]) -> str:
        """构建目标定位提示词"""
        return f"""
你是一个内容分发智能体，负责分析内容并决定最佳的分发策略。

待分发内容：
标题: {content.get('title', '')}
内容: {content.get('content', '')}
类型: {content.get('type', 'post')}

可选用户画像（前10个用户作为示例）：
{json.dumps([user.to_dict() for user in available_users[:10]], ensure_ascii=False, indent=2)}

当前分发策略：
{json.dumps(self.strategy.to_dict() if self.strategy else {}, ensure_ascii=False, indent=2)}

请分析内容特征，结合用户画像和分发策略，推荐最佳的目标用户群体。
考虑因素包括：
1. 内容主题与用户兴趣的匹配度
2. 用户的立场、情感、意图与内容的关联性
3. 用户的活跃度和社会影响力
4. 预期的认知引导效果

请返回JSON格式的分析结果：
{{
    "recommended_users": ["user_id1", "user_id2", ...],
    "targeting_reasoning": "选择这些用户的原因",
    "expected_outcomes": {{
        "engagement_rate": 0.0-1.0之间的预期参与率,
        "sentiment_impact": "预期情感影响",
        "reach_potential": "传播潜力评估"
    }}
}}
"""

    async def analyze_and_recommend_targets(self, content: Dict[str, Any], available_users: List[UserProfile]) -> Dict[str, Any]:
        """分析内容并推荐目标用户"""
        try:
            prompt = self._build_targeting_prompt(content, available_users)

            response = await self.client.chat.completions.acreate(
                model=config.model.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的内容分发分析师，擅长用户画像分析和精准定位。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.model.temperature,
                max_tokens=config.model.max_tokens
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"分发分析时出错: {e}")
            # 返回默认推荐
            return {
                "recommended_users": [user.user_id for user in available_users[:5]],
                "targeting_reasoning": "默认推荐策略",
                "expected_outcomes": {
                    "engagement_rate": 0.3,
                    "sentiment_impact": "中性",
                    "reach_potential": "中等"
                }
            }

    def apply_rule_based_filtering(self, users: List[UserProfile], rules: List[DistributionRule]) -> List[str]:
        """应用基于规则的用户过滤"""
        selected_users = []

        for user in users:
            user_score = 0.0

            for rule in rules:
                if not rule.is_active:
                    continue

                if self._user_matches_rule(user, rule):
                    user_score += rule.weight

            # 根据评分决定是否选择该用户
            if user_score > 0:
                selected_users.append(user.user_id)

        return selected_users

    def _user_matches_rule(self, user: UserProfile, rule: DistributionRule) -> bool:
        """检查用户是否匹配规则"""
        conditions = rule.conditions

        if rule.criteria_type == TargetingCriteria.DEMOGRAPHIC:
            # 人口统计学匹配
            if "age_range" in conditions:
                age_min, age_max = conditions["age_range"]
                if not (age_min <= user.age <= age_max):
                    return False

            if "gender" in conditions:
                if user.gender.value not in conditions["gender"]:
                    return False

            if "occupation" in conditions:
                if user.occupation not in conditions["occupation"]:
                    return False

        elif rule.criteria_type == TargetingCriteria.BEHAVIORAL:
            # 行为特征匹配
            if "activity_level" in conditions:
                min_activity = conditions["activity_level"]
                if user.activity_level < min_activity:
                    return False

            if "social_influence" in conditions:
                min_influence = conditions["social_influence"]
                if user.social_influence < min_influence:
                    return False

        elif rule.criteria_type == TargetingCriteria.PSYCHOGRAPHIC:
            # 心理特征匹配
            if "stance" in conditions:
                if user.stance.value not in conditions["stance"]:
                    return False

            if "emotion" in conditions:
                if user.emotion.value not in conditions["emotion"]:
                    return False

            if "intent" in conditions:
                if user.intent.value not in conditions["intent"]:
                    return False

        return True


class DistributionEvaluator:
    """分发效果评估器"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.model.api_key,
            base_url=config.model.base_url
        )

    def calculate_engagement_metrics(self, environment_data: Dict[str, Any], target_users: List[str]) -> EvaluationMetrics:
        """计算参与度指标"""
        stats = environment_data.get("stats", {})
        comments = environment_data.get("comments", [])

        # 计算参与率
        total_interactions = stats.get("likes", 0) + stats.get("comments", 0) + stats.get("shares", 0)
        engagement_rate = total_interactions / len(target_users) if target_users else 0

        # 计算情感变化（需要情感分析）
        sentiment_shift = self._analyze_sentiment_shift(comments)

        # 计算观点多样性
        opinion_diversity = self._calculate_opinion_diversity(comments)

        # 计算触达率
        participants = set()
        for comment in comments:
            participants.add(comment.get("user_id", ""))
        reach = len(participants)

        # 计算转化率（有意义的交互比例）
        meaningful_interactions = len([c for c in comments if len(c.get("content", "")) > 10])
        conversion_rate = meaningful_interactions / total_interactions if total_interactions > 0 else 0

        # 计算病毒性评分
        virality_score = min(stats.get("shares", 0) / max(1, stats.get("likes", 1)), 1.0)

        return EvaluationMetrics(
            engagement_rate=min(engagement_rate, 1.0),
            sentiment_shift=sentiment_shift,
            opinion_diversity=opinion_diversity,
            reach=reach,
            conversion_rate=conversion_rate,
            time_to_engagement=0.0,  # 需要从时间数据计算
            virality_score=virality_score
        )

    def _analyze_sentiment_shift(self, comments: List[Dict[str, Any]]) -> float:
        """分析情感变化"""
        if not comments:
            return 0.0

        # 简单的情感分析（实际项目中应该使用更复杂的模型）
        positive_words = ["好", "棒", "支持", "赞同", "喜欢", "正确"]
        negative_words = ["差", "坏", "反对", "不对", "讨厌", "错误"]

        sentiment_scores = []
        for comment in comments:
            content = comment.get("content", "")
            pos_count = sum(1 for word in positive_words if word in content)
            neg_count = sum(1 for word in negative_words if word in content)

            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                sentiment_scores.append(sentiment)

        if not sentiment_scores:
            return 0.0

        # 返回情感变化的绝对值作为情感影响力度量
        return abs(sum(sentiment_scores) / len(sentiment_scores))

    def _calculate_opinion_diversity(self, comments: List[Dict[str, Any]]) -> float:
        """计算观点多样性"""
        if len(comments) < 2:
            return 0.0

        # 简单的多样性计算：基于评论长度和内容差异性
        unique_contents = set()
        total_length = 0

        for comment in comments:
            content = comment.get("content", "").strip()
            if content:
                unique_contents.add(content[:50])  # 取前50个字符作为唯一性标识
                total_length += len(content)

        # 多样性 = 唯一内容比例 * 平均长度因子
        uniqueness_ratio = len(unique_contents) / len(comments)
        avg_length_factor = min(total_length / len(comments) / 100, 1.0)  # 归一化到0-1

        return uniqueness_ratio * 0.7 + avg_length_factor * 0.3

    async def evaluate_strategy_performance(self, strategy: DistributionStrategy,
                                            results: List[DistributionResult]) -> Dict[str, Any]:
        """评估策略表现"""
        if not results:
            return {"overall_score": 0.0, "recommendations": ["需要更多数据进行评估"]}

        # 计算平均指标
        avg_metrics = EvaluationMetrics()
        for result in results:
            metrics_dict = result.metrics.to_dict()
            for key, value in metrics_dict.items():
                setattr(avg_metrics, key, getattr(avg_metrics, key) + value)

        # 计算平均值
        num_results = len(results)
        for key in avg_metrics.to_dict().keys():
            setattr(avg_metrics, key, getattr(avg_metrics, key) / num_results)

        # 计算综合评分
        overall_score = avg_metrics.calculate_overall_score()

        # 生成改进建议
        recommendations = await self._generate_improvement_recommendations(strategy, avg_metrics)

        return {
            "overall_score": overall_score,
            "average_metrics": avg_metrics.to_dict(),
            "recommendations": recommendations,
            "total_distributions": num_results
        }

    async def _generate_improvement_recommendations(self, strategy: DistributionStrategy,
                                                    metrics: EvaluationMetrics) -> List[str]:
        """生成改进建议"""
        try:
            prompt = f"""
根据以下分发策略和效果指标，提供改进建议：

当前策略：
{json.dumps(strategy.to_dict(), ensure_ascii=False, indent=2)}

效果指标：
{json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2)}

请分析策略的优缺点，并提供3-5条具体的改进建议。
建议应该具体、可操作，并针对提升认知引导效果。

返回JSON格式：
{{
    "recommendations": ["建议1", "建议2", "建议3", ...]
}}
"""

            response = await self.client.chat.completions.acreate(
                model=config.model.model_name,
                messages=[
                    {"role": "system", "content": "你是一个内容分发策略优化专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("recommendations", [])

        except Exception as e:
            print(f"生成改进建议时出错: {e}")
            return ["建议收集更多数据进行分析", "考虑调整目标用户群体", "优化内容创作策略"]


class DistributionManager:
    """分发管理器"""

    def __init__(self):
        self.strategies: Dict[str, DistributionStrategy] = {}
        self.agent = DistributionAgent()
        self.evaluator = DistributionEvaluator()
        self.results_history: List[DistributionResult] = []

    def add_strategy(self, strategy: DistributionStrategy):
        """添加分发策略"""
        self.strategies[strategy.strategy_id] = strategy

    def get_strategy(self, strategy_id: str) -> Optional[DistributionStrategy]:
        """获取分发策略"""
        return self.strategies.get(strategy_id)

    async def execute_distribution(self, content: Dict[str, Any], available_users: List[UserProfile],
                                   strategy_id: str = None) -> DistributionResult:
        """执行分发"""
        start_time = datetime.now()
        distribution_id = str(uuid.uuid4())

        # 选择策略
        if strategy_id and strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            self.agent.strategy = strategy
        else:
            strategy = self.agent.strategy

        # AI推荐目标用户
        ai_recommendation = await self.agent.analyze_and_recommend_targets(content, available_users)
        ai_recommended_users = ai_recommendation.get("recommended_users", [])

        # 基于规则过滤用户
        if strategy and strategy.rules:
            rule_based_users = self.agent.apply_rule_based_filtering(available_users, strategy.get_active_rules())
            # 合并AI推荐和规则过滤的结果
            target_users = list(set(ai_recommended_users + rule_based_users))
        else:
            target_users = ai_recommended_users

        # 创建分发结果
        result = DistributionResult(
            distribution_id=distribution_id,
            strategy_id=strategy.strategy_id if strategy else "default",
            content_id=content.get("post_id", "unknown"),
            target_users=target_users,
            actual_recipients=target_users,  # 简化实现，假设都成功接收
            metrics=EvaluationMetrics(),  # 初始化空指标，稍后更新
            timestamp=start_time.isoformat(),
            duration=(datetime.now() - start_time).total_seconds()
        )

        self.results_history.append(result)
        return result

    async def evaluate_distribution_effectiveness(self, distribution_id: str,
                                                  environment_data: Dict[str, Any]) -> EvaluationMetrics:
        """评估分发效果"""
        # 找到对应的分发结果
        result = None
        for r in self.results_history:
            if r.distribution_id == distribution_id:
                result = r
                break

        if not result:
            raise ValueError(f"未找到分发ID: {distribution_id}")

        # 计算评估指标
        metrics = self.evaluator.calculate_engagement_metrics(environment_data, result.target_users)

        # 更新结果中的指标
        result.metrics = metrics

        return metrics

    def create_default_strategy(self) -> DistributionStrategy:
        """创建默认分发策略"""
        strategy = DistributionStrategy(
            strategy_id="default_strategy",
            name="默认分发策略",
            description="基于用户活跃度和情感匹配的默认策略",
            created_at=datetime.now().isoformat()
        )

        # 添加一些基本规则
        activity_rule = DistributionRule(
            rule_id="activity_rule",
            name="活跃用户优先",
            description="优先选择活跃度较高的用户",
            criteria_type=TargetingCriteria.BEHAVIORAL,
            conditions={"activity_level": 0.3},
            weight=1.0
        )

        emotion_rule = DistributionRule(
            rule_id="emotion_rule",
            name="情感匹配",
            description="选择情感状态适合的用户",
            criteria_type=TargetingCriteria.PSYCHOGRAPHIC,
            conditions={"emotion": ["neutral", "positive"]},
            weight=0.8
        )

        strategy.add_rule(activity_rule)
        strategy.add_rule(emotion_rule)

        return strategy
