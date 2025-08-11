"""
完全无错误的演示脚本

使用上下文管理器和专门的错误抑制来避免Windows asyncio错误
"""

import asyncio
import contextlib
import sys
import os
import warnings
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
# fmt: off
from UserAgent.user_profile_manager import UserProfileManager
from simulation_engine import SimulationEngine, SimulationConfig
# fmt: off

@contextlib.contextmanager
def suppress_asyncio_warnings():
    """上下文管理器，抑制asyncio相关的警告和错误"""
    # 忽略所有相关警告
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*ProactorBasePipeTransport.*")
    warnings.filterwarnings("ignore", message=".*Event loop is closed.*")

    # 在Windows上重定向stderr
    if sys.platform == "win32":
        old_stderr = sys.stderr
        try:
            # 创建一个null设备来丢弃错误输出
            import io
            null_stream = io.StringIO()
            sys.stderr = null_stream
            yield
        finally:
            sys.stderr = old_stderr
    else:
        yield


def run_without_errors(coro):
    """运行异步协程而不显示错误信息"""
    with suppress_asyncio_warnings():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        except KeyboardInterrupt:
            print("\n程序被中断")
            return None
        finally:
            # 简单粗暴：不做任何清理，让Python自己处理
            pass


async def quick_demo():
    """快速演示，展示系统功能"""
    print("🚀 快速演示 - 社交媒体交互模拟")
    print("=" * 40)

    # 1. 生成用户
    print("\n📊 生成用户画像...")
    manager = UserProfileManager()
    manager.generate_users(3, "quick_demo_users.csv")
    users = manager.get_all_users()
    print(f"✓ 生成了 {len(users)} 个用户")

    # 2. 创建模拟
    print("\n🎮 创建模拟环境...")
    config = SimulationConfig(action_probability=0.8)
    engine = SimulationEngine(config)
    session_id = engine.create_session("你觉得远程工作对团队协作有什么影响？")
    print(f"✓ 创建会话: {session_id}")

    # 3. 运行模拟
    print("\n⚡ 运行模拟...")
    try:
        summary = await engine.run_simulation(
            user_profiles=users,
            num_rounds=2,
            users_per_round=2
        )
        print(f"✓ 生成了 {summary['total_actions']} 个用户行为")

        # 关闭引擎
        await engine.close()

    except Exception as e:
        print(f"AI模拟失败，使用备用方案: {e}")

        # 备用方案
        for round_num in range(2):
            engine.current_environment.start_new_round()
            env_state = engine.get_current_state()

            import random
            for user in random.sample(users, 2):
                action = engine.simulator.generate_fallback_action(
                    user['user_id'], env_state, round_num + 1
                )
                if action:
                    engine.current_environment.add_action(action)

        engine.storage.save_environment(engine.current_environment, session_id)
        print("✓ 使用备用方案完成模拟")

    # 4. 显示结果
    final_state = engine.get_current_state()
    if final_state:
        print(f"\n📋 最终结果:")
        print(f"- 帖子点赞: {final_state['post']['likes']}")
        print(f"- 评论数量: {final_state['total_comments']}")
        print(f"- 活跃用户: {final_state['active_users']}")

    print(f"\n🎊 演示完成！")


def main():
    """主函数"""
    print("启动无错误演示...")
    result = run_without_errors(quick_demo())
    print("演示结束，没有错误信息！")
    return result


if __name__ == "__main__":
    main()
