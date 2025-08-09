"""
用户画像数据结构定义
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import json


class Gender(Enum):
    """性别"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Stance(Enum):
    """立场"""
    SUPPORT = "support"
    OPPOSE = "oppose"
    NEUTRAL = "neutral"


class Emotion(Enum):
    """情感"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Intent(Enum):
    """意图"""
    SHARE = "share"
    DISCUSS = "discuss"
    CRITICIZE = "criticize"
    SUPPORT = "support"
    IGNORE = "ignore"


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str

    # 基本社会特征
    age: int
    gender: Gender
    occupation: str
    education_level: str
    location: str

    # 针对事件的态度
    stance: Stance  # 立场
    emotion: Emotion  # 情感
    intent: Intent  # 意图

    # 个性化特征
    personality_traits: Dict[str, float] = field(default_factory=dict)  # 例如：开放性、外向性等
    interests: List[str] = field(default_factory=list)
    social_influence: float = 0.5  # 社会影响力，0-1之间

    # 行为偏好
    activity_level: float = 0.5  # 活跃度，0-1之间
    interaction_preference: List[str] = field(default_factory=list)  # 偏好的交互方式

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "user_id": self.user_id,
            "age": self.age,
            "gender": self.gender.value if isinstance(self.gender, Gender) else self.gender,
            "occupation": self.occupation,
            "education_level": self.education_level,
            "location": self.location,
            "stance": self.stance.value if isinstance(self.stance, Stance) else self.stance,
            "emotion": self.emotion.value if isinstance(self.emotion, Emotion) else self.emotion,
            "intent": self.intent.value if isinstance(self.intent, Intent) else self.intent,
            "personality_traits": self.personality_traits,
            "interests": self.interests,
            "social_influence": self.social_influence,
            "activity_level": self.activity_level,
            "interaction_preference": self.interaction_preference
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """从字典创建用户画像"""
        return cls(
            user_id=data["user_id"],
            age=data["age"],
            gender=Gender(data["gender"]) if isinstance(data["gender"], str) else data["gender"],
            occupation=data["occupation"],
            education_level=data["education_level"],
            location=data["location"],
            stance=Stance(data["stance"]) if isinstance(data["stance"], str) else data["stance"],
            emotion=Emotion(data["emotion"]) if isinstance(data["emotion"], str) else data["emotion"],
            intent=Intent(data["intent"]) if isinstance(data["intent"], str) else data["intent"],
            personality_traits=data.get("personality_traits", {}),
            interests=data.get("interests", []),
            social_influence=data.get("social_influence", 0.5),
            activity_level=data.get("activity_level", 0.5),
            interaction_preference=data.get("interaction_preference", [])
        )


@dataclass
class UserMemory:
    """用户记忆"""
    user_id: str
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    max_length: int = 5

    def add_interaction(self, interaction: Dict[str, Any]):
        """添加交互记录"""
        self.interactions.append(interaction)
        if len(self.interactions) > self.max_length:
            self.interactions.pop(0)  # 移除最老的记录

    def get_recent_interactions(self, count: int = None) -> List[Dict[str, Any]]:
        """获取最近的交互记录"""
        if count is None:
            return self.interactions.copy()
        return self.interactions[-count:] if count <= len(self.interactions) else self.interactions.copy()

    def clear(self):
        """清空记忆"""
        self.interactions.clear()
