"""
用户画像工具

简化的命令行工具，用于生成和管理用户画像
"""

import argparse
import os
import sys

# 添加项目根目录到路径 - 必须在导入之前
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# fmt: off
from UserAgent import UserProfileGenerator, UserProfileManager  # noqa: E402
# fmt: on


def generate_users(count: int, output_file: str = None) -> None:
    """生成用户画像"""
    print(f"正在生成 {count} 个用户画像...")

    manager = UserProfileManager()
    filepath = manager.generate_users(count, output_file)

    print(f"成功生成 {count} 个用户")
    print(f"保存路径: {filepath}")

    # 显示前几个用户的示例
    users = manager.get_all_users()
    print(f"\n前2个用户示例:")
    for i, user in enumerate(users[:2]):
        print(f"\n用户 {i+1}: {user['user_id']}")
        for key, value in user.items():
            if key not in ['user_id', 'created_at']:
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(value)}")
                else:
                    print(f"  {key}: {value}")


def show_stats(filename: str) -> None:
    """显示用户统计"""
    print(f"正在分析用户文件: {filename}")

    manager = UserProfileManager()
    try:
        count = manager.load_users_from_file(filename)
        print(f"成功加载 {count} 个用户")

        # 生成报告
        report = manager.generate_report()
        print(report)

    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
    except Exception as e:
        print(f"错误: {e}")


def list_files() -> None:
    """列出所有用户文件"""
    manager = UserProfileManager()
    files = manager.list_saved_files()

    if files:
        print("已保存的用户文件:")
        for i, filename in enumerate(files, 1):
            print(f"  {i}. {filename}")
    else:
        print("没有找到已保存的用户文件")


def delete_user(filename: str, user_id: str) -> None:
    """删除指定用户"""
    manager = UserProfileManager()
    try:
        count = manager.load_users_from_file(filename)
        print(f"加载了 {count} 个用户")

        success = manager.delete_user(user_id)
        if success:
            print(f"成功删除用户: {user_id}")
            # 保存修改后的文件
            new_filepath = manager.save_current_users(filename)
            print(f"文件已更新: {new_filepath}")
            print(f"剩余用户数: {len(manager.get_all_users())}")
        else:
            print(f"未找到用户: {user_id}")

    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
    except Exception as e:
        print(f"错误: {e}")


def show_config() -> None:
    """显示配置信息"""
    generator = UserProfileGenerator()
    attributes = generator.get_available_attributes()

    print("当前配置的用户属性:")
    for attr_name, attr_values in attributes.items():
        print(f"\n{attr_name}:")
        if isinstance(attr_values, list) and len(attr_values) <= 10:
            for value in attr_values:
                print(f"  - {value}")
        else:
            print(f"  共 {len(attr_values)} 个选项")


def main():
    parser = argparse.ArgumentParser(description='用户画像工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 生成用户命令
    generate_parser = subparsers.add_parser('generate', help='生成用户画像')
    generate_parser.add_argument('count', type=int, help='生成用户数量')
    generate_parser.add_argument('-o', '--output', help='输出文件名')

    # 显示统计命令
    stats_parser = subparsers.add_parser('stats', help='显示用户统计')
    stats_parser.add_argument('filename', help='用户文件名')

    # 列出文件命令
    subparsers.add_parser('list', help='列出所有用户文件')

    # 删除用户命令
    delete_parser = subparsers.add_parser('delete', help='删除指定用户')
    delete_parser.add_argument('filename', help='用户文件名')
    delete_parser.add_argument('user_id', help='要删除的用户ID')

    # 显示配置命令
    subparsers.add_parser('config', help='显示配置信息')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_users(args.count, args.output)
    elif args.command == 'stats':
        show_stats(args.filename)
    elif args.command == 'list':
        list_files()
    elif args.command == 'delete':
        delete_user(args.filename, args.user_id)
    elif args.command == 'config':
        show_config()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
