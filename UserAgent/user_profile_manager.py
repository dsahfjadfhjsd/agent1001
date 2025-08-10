"""
用户画像管理器

管理已生成的用户画像，提供基本的读取、统计、删除功能
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from collections import Counter
from datetime import datetime

# 处理相对导入问题
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from user_profile_generator import UserProfileGenerator
else:
    from .user_profile_generator import UserProfileGenerator


class UserProfileManager:
    """用户画像管理器"""

    def __init__(self, generator: UserProfileGenerator = None):
        """
        初始化管理器

        Args:
            generator: 用户画像生成器实例
        """
        self.generator = generator or UserProfileGenerator()
        self.users_cache = []  # 缓存加载的用户数据
        self.current_file = None  # 当前加载的文件名

    def generate_users(self, count: int, filename: str = None) -> str:
        """
        生成用户并保存

        Args:
            count: 生成用户数量
            filename: 保存文件名

        Returns:
            保存的文件路径
        """
        users = self.generator.generate_multiple_users(count)
        filepath = self.generator.save_users_to_csv(users, filename)
        self.users_cache = users
        self.current_file = filename or os.path.basename(filepath)
        return filepath

    def load_users_from_file(self, filename: str) -> int:
        """
        从文件加载用户数据到缓存

        Args:
            filename: CSV文件名

        Returns:
            加载的用户数量
        """
        self.users_cache = self.generator.load_users_from_csv(filename)
        self.current_file = filename
        return len(self.users_cache)

    def get_all_users(self) -> List[Dict[str, Any]]:
        """获取所有用户"""
        return self.users_cache.copy()

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据用户ID获取用户"""
        for user in self.users_cache:
            if user.get('user_id') == user_id:
                return user
        return None

    def get_user_statistics(self) -> Dict[str, Any]:
        """获取用户统计信息"""
        if not self.users_cache:
            return {}

        stats = {
            'total_users': len(self.users_cache),
            'demographics': {},
            'behavior': {}
        }

        # 统计人口学特征
        demographic_fields = ['age_group', 'gender', 'occupation', 'education', 'location']
        for field in demographic_fields:
            if any(field in user for user in self.users_cache):
                field_counter = Counter(user.get(field, 'Unknown') for user in self.users_cache)
                stats['demographics'][field] = dict(field_counter)

        # 统计行为特征
        behavior_fields = ['activity_level', 'stance', 'sentiment', 'intent']
        for field in behavior_fields:
            if any(field in user for user in self.users_cache):
                field_counter = Counter(user.get(field, 'Unknown') for user in self.users_cache)
                stats['behavior'][field] = dict(field_counter)

        return stats

    def generate_report(self) -> str:
        """生成简单的统计报告"""
        if not self.users_cache:
            return "没有用户数据"

        stats = self.get_user_statistics()

        report = []
        report.append("=" * 40)
        report.append("用户画像统计报告")
        report.append("=" * 40)
        report.append(f"总用户数: {stats['total_users']}")
        report.append("")

        # 人口学特征
        report.append("【人口学特征】")
        for field, distribution in stats['demographics'].items():
            report.append(f"{field}: {len(distribution)} 个不同值")
            # 显示前3个最多的
            sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            for value, count in sorted_items[:3]:
                percentage = (count / stats['total_users']) * 100
                report.append(f"  {value}: {count} ({percentage:.1f}%)")

        report.append("")
        # 行为特征
        report.append("【行为特征】")
        for field, distribution in stats['behavior'].items():
            report.append(f"{field}: {len(distribution)} 个不同值")
            # 显示前3个最多的
            sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            for value, count in sorted_items[:3]:
                percentage = (count / stats['total_users']) * 100
                report.append(f"  {value}: {count} ({percentage:.1f}%)")

        return "\n".join(report)

    def delete_user(self, user_id: str) -> bool:
        """
        删除指定用户

        Args:
            user_id: 用户ID

        Returns:
            是否删除成功
        """
        for i, user in enumerate(self.users_cache):
            if user.get('user_id') == user_id:
                del self.users_cache[i]
                return True
        return False

    def save_current_users(self, filename: str = None) -> str:
        """
        保存当前缓存中的用户到文件

        Args:
            filename: 文件名

        Returns:
            保存的文件路径
        """
        if not self.users_cache:
            raise ValueError("没有用户数据可保存")

        filepath = self.generator.save_users_to_csv(self.users_cache, filename)
        if filename:
            self.current_file = filename
        return filepath

    def list_saved_files(self) -> List[str]:
        """列出所有已保存的用户文件"""
        return self.generator.list_saved_user_files()


# 示例使用
if __name__ == "__main__":
    # 创建管理器
    manager = UserProfileManager()

    # 生成用户
    filepath = manager.generate_users(10, "test_users.csv")
    print(f"生成了 10 个用户，保存到: {filepath}")

    # 生成统计报告
    print("\n" + manager.generate_report())

    # 删除一个用户
    users = manager.get_all_users()
    if users:
        first_user_id = users[0]['user_id']
        success = manager.delete_user(first_user_id)
        print(f"\n删除用户 {first_user_id}: {'成功' if success else '失败'}")
        print(f"剩余用户数: {len(manager.get_all_users())}")

    # 列出文件
    print(f"\n已保存的文件: {manager.list_saved_files()}")
