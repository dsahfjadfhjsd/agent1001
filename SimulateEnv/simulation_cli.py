"""
交互模拟命令行接口

提供简单易用的命令行界面来运行社交媒体交互模拟
"""

from UserAgent.user_profile_manager import UserProfileManager
from SimulateEnv.simulation_engine import SimulationEngine, SimulationConfig
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class SimulationCLI:
    """模拟命令行界面"""

    def __init__(self):
        self.engine = None
        self.user_manager = UserProfileManager()
        self.current_users = []

    def print_header(self):
        """打印程序头部"""
        print("="*60)
        print("           社交媒体交互模拟器")
        print("="*60)
        print()

    def print_menu(self):
        """打印主菜单"""
        print("\n" + "-"*40)
        print("主菜单:")
        print("1. 创建新的模拟会话")
        print("2. 加载已有会话")
        print("3. 加载用户画像")
        print("4. 运行单轮模拟")
        print("5. 运行多轮模拟")
        print("6. 查看当前状态")
        print("7. 查看会话列表")
        print("8. 导出会话数据")
        print("9. 配置设置")
        print("0. 退出")
        print("-"*40)

    def get_user_input(self, prompt: str, default: str = None) -> str:
        """获取用户输入"""
        if default:
            full_prompt = f"{prompt} (默认: {default}): "
        else:
            full_prompt = f"{prompt}: "

        result = input(full_prompt).strip()
        if not result and default:
            return default
        return result

    def get_int_input(self, prompt: str, default: int = None, min_val: int = None, max_val: int = None) -> int:
        """获取整数输入"""
        while True:
            try:
                if default:
                    full_prompt = f"{prompt} (默认: {default}): "
                else:
                    full_prompt = f"{prompt}: "

                result = input(full_prompt).strip()
                if not result and default is not None:
                    result = str(default)

                value = int(result)

                if min_val is not None and value < min_val:
                    print(f"输入值不能小于 {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"输入值不能大于 {max_val}")
                    continue

                return value
            except ValueError:
                print("请输入有效的整数")

    def create_session(self):
        """创建新会话"""
        print("\n创建新的模拟会话")
        print("-" * 20)

        # 获取帖子内容
        post_content = self.get_user_input("请输入初始帖子内容")
        if not post_content:
            print("帖子内容不能为空")
            return

        # 获取会话ID（可选）
        session_id = self.get_user_input("请输入会话ID（可选，留空自动生成）")

        try:
            # 创建引擎和会话
            config = SimulationConfig()
            self.engine = SimulationEngine(config)

            if session_id:
                created_session_id = self.engine.create_session(post_content, session_id)
            else:
                created_session_id = self.engine.create_session(post_content)

            print(f"\n✓ 成功创建会话: {created_session_id}")

        except Exception as e:
            print(f"✗ 创建会话失败: {e}")

    def load_session(self):
        """加载已有会话"""
        print("\n加载已有会话")
        print("-" * 15)

        # 创建引擎实例来列出会话
        if not self.engine:
            config = SimulationConfig()
            self.engine = SimulationEngine(config)

        # 显示可用会话
        sessions = self.engine.list_all_sessions()
        if not sessions:
            print("没有找到任何会话")
            return

        print("可用会话:")
        for i, session_id in enumerate(sessions, 1):
            summary = self.engine.get_session_summary(session_id)
            if summary:
                print(f"{i}. {session_id} (轮数: {summary.get('total_rounds', 0)}, 行为数: {summary.get('total_actions', 0)})")
            else:
                print(f"{i}. {session_id}")

        # 获取用户选择
        try:
            choice = self.get_int_input("请选择会话序号", min_val=1, max_val=len(sessions))
            session_id = sessions[choice - 1]

            if self.engine.load_session(session_id):
                print(f"✓ 成功加载会话: {session_id}")
            else:
                print(f"✗ 加载会话失败: {session_id}")

        except (ValueError, IndexError):
            print("无效的选择")

    def load_users(self):
        """加载用户画像"""
        print("\n加载用户画像")
        print("-" * 12)

        # 显示可用的用户文件
        user_files = self.user_manager.list_saved_files()
        if not user_files:
            print("没有找到用户画像文件")

            # 提供生成新用户的选项
            generate = self.get_user_input("是否生成新的用户画像？(y/n)", "n")
            if generate.lower() == 'y':
                self.generate_users()
            return

        print("可用用户文件:")
        for i, filename in enumerate(user_files, 1):
            print(f"{i}. {filename}")

        try:
            choice = self.get_int_input("请选择文件序号", min_val=1, max_val=len(user_files))
            filename = user_files[choice - 1]

            count = self.user_manager.load_users_from_file(filename)
            self.current_users = self.user_manager.get_all_users()
            print(f"✓ 成功加载 {count} 个用户画像")

        except (ValueError, IndexError):
            print("无效的选择")

    def generate_users(self):
        """生成新用户"""
        print("\n生成新用户画像")
        print("-" * 14)

        count = self.get_int_input("请输入要生成的用户数量", 10, 1, 100)
        filename = self.get_user_input("请输入保存文件名（不含扩展名）", f"users_{count}")

        try:
            filepath = self.user_manager.generate_users(count, f"{filename}.csv")
            self.current_users = self.user_manager.get_all_users()
            print(f"✓ 成功生成 {count} 个用户，保存到: {filepath}")

        except Exception as e:
            print(f"✗ 生成用户失败: {e}")

    async def run_single_round(self):
        """运行单轮模拟"""
        if not self.engine:
            print("请先创建或加载会话")
            return

        if not self.current_users:
            print("请先加载用户画像")
            return

        print(f"\n运行单轮模拟")
        print("-" * 12)

        # 选择参与用户数
        max_users = len(self.current_users)
        user_count = self.get_int_input(
            f"请输入参与用户数 (最大 {max_users})",
            min(5, max_users),
            1,
            max_users
        )

        # 随机选择用户
        import random
        selected_users = random.sample(self.current_users, user_count)

        try:
            print("\n开始模拟...")
            actions = await self.engine.simulate_round(selected_users)
            print(f"✓ 模拟完成，生成了 {len(actions)} 个用户行为")

        except Exception as e:
            print(f"✗ 模拟失败: {e}")

    async def run_multi_round(self):
        """运行多轮模拟"""
        if not self.engine:
            print("请先创建或加载会话")
            return

        if not self.current_users:
            print("请先加载用户画像")
            return

        print(f"\n运行多轮模拟")
        print("-" * 12)

        # 获取参数
        num_rounds = self.get_int_input("请输入模拟轮数", 3, 1, 20)

        max_users = len(self.current_users)
        users_per_round = self.get_int_input(
            f"请输入每轮参与用户数 (最大 {max_users})",
            min(5, max_users),
            1,
            max_users
        )

        randomize = self.get_user_input("是否随机选择用户？(y/n)", "y").lower() == 'y'

        try:
            print(f"\n开始 {num_rounds} 轮模拟...")
            summary = await self.engine.run_simulation(
                user_profiles=self.current_users,
                num_rounds=num_rounds,
                users_per_round=users_per_round,
                randomize_users=randomize
            )

            print(f"\n✓ 多轮模拟完成！")
            print(f"总行为数: {summary['total_actions']}")
            print(f"参与用户数: {summary['unique_users']}")

        except Exception as e:
            print(f"✗ 模拟失败: {e}")

    def view_current_state(self):
        """查看当前状态"""
        if not self.engine:
            print("没有活跃会话")
            return

        state = self.engine.get_current_state()
        if not state:
            print("没有当前状态")
            return

        print(f"\n当前环境状态")
        print("-" * 14)

        # 显示帖子信息
        post = state['post']
        print(f"帖子内容: {post['content']}")
        print(f"帖子点赞数: {post['likes']}")

        # 显示评论统计
        print(f"总评论数: {state['total_comments']}")
        print(f"一级评论数: {len(state['primary_comments'])}")

        # 显示部分评论
        if state['primary_comments']:
            print("\n最新评论:")
            for i, comment in enumerate(state['primary_comments'][-3:], 1):  # 显示最后3条
                print(f"{i}. {comment['content'][:50]}... (点赞: {comment['likes']})")
                if comment.get('sub_comments'):
                    print(f"   └ 有 {len(comment['sub_comments'])} 条回复")

        print(f"\n交互统计:")
        print(f"当前轮数: {state['current_round']}")
        print(f"总行为数: {state['total_actions']}")
        print(f"活跃用户数: {state['active_users']}")

    def view_sessions(self):
        """查看会话列表"""
        print(f"\n会话列表")
        print("-" * 8)

        if not self.engine:
            config = SimulationConfig()
            self.engine = SimulationEngine(config)

        sessions = self.engine.list_all_sessions()
        if not sessions:
            print("没有找到任何会话")
            return

        for session_id in sessions:
            summary = self.engine.get_session_summary(session_id)
            if summary:
                print(f"\n会话: {session_id}")
                print(f"  创建时间: {summary.get('created_at', '未知')}")
                print(f"  轮数: {summary.get('total_rounds', 0)}")
                print(f"  行为数: {summary.get('total_actions', 0)}")
                print(f"  用户数: {summary.get('unique_users', 0)}")

    def export_data(self):
        """导出数据"""
        if not self.engine:
            print("没有活跃会话")
            return

        print(f"\n导出会话数据")
        print("-" * 12)

        try:
            export_dir = self.engine.export_session()
            print(f"✓ 数据已导出到: {export_dir}")

        except Exception as e:
            print(f"✗ 导出失败: {e}")

    def configure_settings(self):
        """配置设置"""
        print(f"\n配置设置")
        print("-" * 8)

        print("当前配置:")
        if self.engine and self.engine.config:
            config = self.engine.config
            print(f"最大并发请求数: {config.max_concurrent_requests}")
            print(f"请求超时时间: {config.request_timeout}秒")
            print(f"模型名称: {config.model_name}")
            print(f"用户行动概率: {config.action_probability}")
        else:
            print("使用默认配置")

        modify = self.get_user_input("是否修改配置？(y/n)", "n")
        if modify.lower() == 'y':
            # 可以添加配置修改逻辑
            print("配置修改功能待实现")

    async def run(self):
        """运行主程序"""
        self.print_header()

        while True:
            self.print_menu()

            try:
                choice = input("请选择操作 (0-9): ").strip()

                if choice == '0':
                    print("感谢使用！")
                    break
                elif choice == '1':
                    self.create_session()
                elif choice == '2':
                    self.load_session()
                elif choice == '3':
                    self.load_users()
                elif choice == '4':
                    await self.run_single_round()
                elif choice == '5':
                    await self.run_multi_round()
                elif choice == '6':
                    self.view_current_state()
                elif choice == '7':
                    self.view_sessions()
                elif choice == '8':
                    self.export_data()
                elif choice == '9':
                    self.configure_settings()
                else:
                    print("无效的选择，请重试")

                # 等待用户按键继续
                if choice != '0':
                    input("\n按回车键继续...")

            except KeyboardInterrupt:
                print("\n\n程序被中断")
                break
            except Exception as e:
                print(f"\n操作出错: {e}")
                input("按回车键继续...")


async def main():
    """主函数"""
    cli = SimulationCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
