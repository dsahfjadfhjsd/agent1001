"""
用户画像生成器

根据配置文件中的属性规则生成用户画像
支持灵活的属性配置和扩展
"""

import json
import random
import uuid
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
from pathlib import Path


class UserProfileGenerator:
    """用户画像生成器"""

    def __init__(self, config_path: str = None):
        """
        初始化生成器

        Args:
            config_path: 配置文件路径，默认使用 Config/user_profiles.json
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                       'Config', 'user_profiles.json')

        self.config_path = config_path
        self.config = self._load_config()
        self.generated_users = []

        # 创建用户存储目录
        self.storage_dir = os.path.join(os.path.dirname(__file__), 'generated_users')
        os.makedirs(self.storage_dir, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"配置文件格式错误: {self.config_path}")

    def _generate_unique_id(self) -> str:
        """生成唯一用户ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_suffix = str(uuid.uuid4())[:8]
        return f"USER_{timestamp}_{unique_suffix}"

    def _select_from_distribution(self, distribution: Dict[str, float]) -> str:
        """根据概率分布选择值"""
        choices = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(choices, weights=weights)[0]

    def _select_from_list(self, items: List[str]) -> str:
        """从列表中随机选择一个值"""
        return random.choice(items)

    def _generate_basic_profile(self) -> Dict[str, Any]:
        """生成基本用户画像"""
        generation_rules = self.config.get('generation_rules', {})

        profile = {
            'user_id': self._generate_unique_id(),
            'created_at': datetime.now().isoformat()
        }

        # 年龄
        if 'age_distribution' in generation_rules:
            profile['age_group'] = self._select_from_distribution(
                generation_rules['age_distribution']
            )

        # 性别
        if 'gender_distribution' in generation_rules:
            profile['gender'] = self._select_from_distribution(
                generation_rules['gender_distribution']
            )

        # 职业
        if 'occupation_list' in generation_rules:
            profile['occupation'] = self._select_from_list(
                generation_rules['occupation_list']
            )

        # 教育水平
        if 'education_levels' in generation_rules:
            profile['education'] = self._select_from_list(
                generation_rules['education_levels']
            )

        # 地区
        if 'locations' in generation_rules:
            profile['location'] = self._select_from_list(
                generation_rules['locations']
            )

        return profile

    def _generate_behavior_profile(self) -> Dict[str, Any]:
        """生成行为画像"""
        behavior_patterns = self.config.get('behavior_patterns', {})
        behavior_profile = {}

        # 活跃度
        if 'activity_levels' in behavior_patterns:
            behavior_profile['activity_level'] = self._select_from_distribution(
                behavior_patterns['activity_levels']
            )

        # 立场
        if 'stance_distribution' in behavior_patterns:
            stance_data = behavior_patterns['stance_distribution']
            stances = ['支持', '中立', '反对']
            probabilities = stance_data.get('probability', [0.33, 0.33, 0.34])
            stance = random.choices(stances, weights=probabilities)[0]
            behavior_profile['stance'] = stance
            # 对应关键词中挑选
            if stance in stance_data:
                behavior_profile['stance_keywords'] = random.sample(stance_data[stance], k=1)

        # 情感倾向
        if 'sentiment_distribution' in behavior_patterns:
            sentiment_data = behavior_patterns['sentiment_distribution']
            sentiments = ['积极', '中立', '消极']
            probabilities = sentiment_data.get('probability', [0.33, 0.33, 0.34])
            sentiment = random.choices(sentiments, weights=probabilities)[0]
            behavior_profile['sentiment'] = sentiment
            # 选择对应的关键词
            if sentiment in sentiment_data:
                behavior_profile['sentiment_keywords'] = random.sample(sentiment_data[sentiment], k=1)

        # 意图
        if 'intent_distribution' in behavior_patterns:
            intent_data = behavior_patterns['intent_distribution']
            intents = ['信息验证', '情感表达', '利益实践']
            probabilities = intent_data.get('probability', [0.33, 0.33, 0.34])
            intent = random.choices(intents, weights=probabilities)[0]
            behavior_profile['intent'] = intent
            # 选择对应的关键词
            if intent in intent_data:
                behavior_profile['intent_keywords'] = random.sample(intent_data[intent], k=1)

        return behavior_profile

    def generate_user_profile(self) -> Dict[str, Any]:
        """生成完整的用户画像"""
        # 生成基本画像
        profile = self._generate_basic_profile()

        # 生成行为画像
        behavior_profile = self._generate_behavior_profile()

        # 合并画像
        profile.update(behavior_profile)

        return profile

    def generate_multiple_users(self, count: int) -> List[Dict[str, Any]]:
        """生成多个用户画像"""
        users = []
        for _ in range(count):
            user = self.generate_user_profile()
            users.append(user)
            self.generated_users.append(user)

        return users

    def save_users_to_csv(self, users: List[Dict[str, Any]], filename: str = None) -> str:
        """将用户画像保存到CSV文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"users_{timestamp}.csv"

        filepath = os.path.join(self.storage_dir, filename)

        if not users:
            raise ValueError("没有用户数据可保存")

        # 获取所有可能的字段
        all_fields = set()
        for user in users:
            all_fields.update(user.keys())

        # 对字段进行排序，保证一致性
        fieldnames = sorted(list(all_fields))

        # id字段放在最前面
        fieldnames.remove('user_id')
        fieldnames = ['user_id'] + fieldnames

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for user in users:
                # 处理列表类型的字段（如关键词）
                processed_user = {}
                for key, value in user.items():
                    if isinstance(value, list):
                        processed_user[key] = '|'.join(value)
                    else:
                        processed_user[key] = value
                writer.writerow(processed_user)

        return filepath

    def load_users_from_csv(self, filename: str) -> List[Dict[str, Any]]:
        """从CSV文件加载用户画像"""
        filepath = os.path.join(self.storage_dir, filename)

        users = []
        with open(filepath, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 处理列表类型的字段
                processed_user = {}
                for key, value in row.items():
                    if key.endswith('_keywords') and value:
                        processed_user[key] = value.split('|')
                    else:
                        processed_user[key] = value
                users.append(processed_user)

        return users

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据用户ID获取用户画像"""
        for user in self.generated_users:
            if user.get('user_id') == user_id:
                return user
        return None

    def get_available_attributes(self) -> Dict[str, List[str]]:
        """获取所有可用的属性配置"""
        attributes = {}

        generation_rules = self.config.get('generation_rules', {})
        for key, value in generation_rules.items():
            if isinstance(value, dict):
                attributes[key] = list(value.keys())
            elif isinstance(value, list):
                attributes[key] = value

        behavior_patterns = self.config.get('behavior_patterns', {})
        for key, value in behavior_patterns.items():
            if isinstance(value, dict) and key != 'stance_distribution' and key != 'sentiment_distribution' and key != 'intent_distribution':
                attributes[key] = list(value.keys())

        return attributes

    def list_saved_user_files(self) -> List[str]:
        """列出所有已保存的用户文件"""
        return [f for f in os.listdir(self.storage_dir) if f.endswith('.csv')]


# 示例使用
if __name__ == "__main__":
    # 创建生成器
    generator = UserProfileGenerator()

    # 生成单个用户
    print("=== 生成单个用户画像 ===")
    user = generator.generate_user_profile()
    print(f"用户ID: {user['user_id']}")
    for key, value in user.items():
        if key != 'user_id':
            print(f"{key}: {value}")

    print("\n=== 生成多个用户画像 ===")
    # 生成多个用户
    users = generator.generate_multiple_users(5)
    print(f"成功生成 {len(users)} 个用户")

    # 保存到CSV
    filepath = generator.save_users_to_csv(users)
    print(f"用户画像已保存到: {filepath}")

    # 显示可用属性
    print("\n=== 可用属性配置 ===")
    attributes = generator.get_available_attributes()
    for attr_name, attr_values in attributes.items():
        print(f"{attr_name}: {attr_values}")
