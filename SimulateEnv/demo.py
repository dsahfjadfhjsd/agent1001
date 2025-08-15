"""
演示脚本 - 展示交互模拟系统的完整功能

这个脚本展示了如何使用交互模拟系统
"""


import asyncio
import json
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
# fmt:off
from SimulateEnv import SimulationEngine, SimulationConfig
from UserAgent.user_profile_manager import UserProfileManager
# fmt:on


async def demo_complete_simulation():
    """完整的模拟演示"""
    print("🚀 社交媒体交互模拟系统演示")
    print("=" * 50)

    # 1. 创建用户画像
    print("\n📊 步骤1: 创建用户画像")
    print("-" * 25)

    manager = UserProfileManager()

    # 生成演示用户
    filepath = manager.generate_users(8, "demo_users.csv")
    users = manager.get_all_users()
    print(f"✓ 生成了 {len(users)} 个用户画像，保存到: {filepath}")

    # 显示用户信息
    print("\n用户画像样例：")
    for i, user in enumerate(users[:3], 1):
        print(f"{i}. {user['user_id'][:15]}... - {user.get('occupation', '未知')} - {user.get('stance', '中立')}")

    # 2. 创建模拟环境
    print(f"\n🏗️  步骤2: 创建模拟环境")
    print("-" * 25)

    config = SimulationConfig(
        max_concurrent_requests=5,
        action_probability=1.0,  # 100%的用户会采取行动
        comment_probability=0.6   # 60%的行动是评论
    )

    engine = SimulationEngine(config)

    # 创建会话
    post_content = "前两天高速上，小米su7车上3个女生出事的动画复盘。感觉是疲劳驾驶睡着了？自动驾驶可以短暂用一下，比如打了哈欠，拿纸巾擦擦眼。还是别长时间交给自动驾驶比如睡觉开车。 #小米##小米su7##小米su7高速碰撞爆燃事件细节#"
    session_id = engine.create_session(post_content)
    # session_id = engine.create_session(post_content, session_id='demo_session_2')

    print(f"✓ 创建会话: {session_id}")
    print(f"✓ 初始帖子: {post_content}")

    # 3. 运行多轮模拟
    print(f"\n🎮 步骤3: 运行多轮交互模拟")
    print("-" * 30)

    try:
        summary = await engine.run_simulation(
            user_profiles=users,
            num_rounds=5,  # 3轮模拟
            users_per_round=5,  # 每轮5个用户
            randomize_users=True
        )

        print(f"\n🎉 模拟完成！")
        print(f"总结果:")
        print(f"- 总行为数: {summary['total_actions']}")
        print(f"- 参与用户数: {summary['unique_users']}")
        print(f"- 行为类型分布: {summary['action_types']}")

    except Exception as e:
        print(f"AI模拟失败: {e}")
        print("使用备用模拟方案...")

        # 使用备用方案
        for round_num in range(3):
            engine.current_environment.start_new_round()

            # 随机选择用户
            import random
            round_users = random.sample(users, 5)

            # 生成备用行为
            env_state = engine.get_current_state()
            actions = []

            for user in round_users:
                action = engine.simulator.generate_fallback_action(
                    user['user_id'], env_state, round_num + 1
                )
                if action and engine.current_environment.add_action(action):
                    actions.append(action)

            print(f"第{round_num + 1}轮: 生成 {len(actions)} 个行为")

        # 保存结果
        engine.storage.save_environment(engine.current_environment, session_id)

    # 4. 查看最终结果
    print(f"\n📋 步骤4: 查看最终环境状态")
    print("-" * 30)

    final_state = engine.get_current_state()
    if final_state:
        post = final_state['post']
        comments = final_state['primary_comments']

        print(f"帖子状态:")
        print(f"- 内容: {post['content'][:50]}...")
        print(f"- 点赞数: {post['likes']}")
        print(f"- 评论数: {final_state['total_comments']}")

        if comments:
            print(f"\n评论展示 (前3条):")
            for i, comment in enumerate(comments[:3], 1):
                print(f"{i}. 【{comment['author_id'][:10]}...】: {comment['content'][:60]}...")
                print(f"   💗 {comment['likes']} 个赞")

                # 显示回复
                if comment.get('sub_comments'):
                    for sub in comment['sub_comments'][:2]:  # 最多显示2条回复
                        print(f"   └ 【{sub['author_id'][:10]}...】: {sub['content'][:40]}...")

    # 5. 导出数据
    print(f"\n💾 步骤5: 导出数据")
    print("-" * 20)

    try:
        export_path = engine.export_session()
        print(f"✓ 数据已导出到: {export_path}")

        # 显示导出文件
        export_dir = Path(export_path)
        if export_dir.exists():
            files = list(export_dir.glob("*.csv")) + list(export_dir.glob("*.json"))
            print(f"导出文件:")
            for file in files:
                print(f"- {file.name}")

    except Exception as e:
        print(f"导出失败: {e}")

    # 关闭引擎
    await engine.close()

    print(f"\n🎊 演示完成！")
    print("=" * 50)

    return session_id


async def demo_api_usage():
    """API使用演示"""
    print("\n💻 API使用演示")
    print("-" * 15)

    # 演示如何通过API使用系统
    from SimulateEnv import InteractionEnvironment, UserAction, ActionType, DataStorage

    print("1. 创建环境")
    env = InteractionEnvironment("这是一个API演示帖子")
    print(f"   帖子ID: {env.post.post_id}")

    print("2. 添加用户行为")
    # 点赞
    like_action = UserAction("", "demo_user_1", ActionType.LIKE_POST, env.post.post_id)
    env.add_action(like_action)
    print(f"   ✓ 用户点赞")

    # 评论
    comment_action = UserAction("", "demo_user_2", ActionType.COMMENT_POST, env.post.post_id, "很棒的内容！")
    env.add_action(comment_action)
    print(f"   ✓ 用户评论")

    print("3. 获取环境状态")
    state = env.get_environment_state()
    print(f"   点赞数: {state['post']['likes']}")
    print(f"   评论数: {state['total_comments']}")

    print("4. 保存数据")
    storage = DataStorage()
    storage.save_environment(env, "demo_session")
    print(f"   ✓ 数据已保存")

    print("API演示完成！")


if __name__ == "__main__":
    async def main():
        await demo_complete_simulation()
        # await demo_api_usage()

    # 使用更安全的运行方式
    import os
    import sys
    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)

    from async_utils import run_async_simple
    run_async_simple(main())
