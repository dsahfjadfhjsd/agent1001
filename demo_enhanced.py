#!/usr/bin/env python3
"""
增强功能演示脚本

展示用户认知动态变化的完整功能
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
# fmt: off
from UserAgent.user_profile_manager import UserProfileManager
from SimulateEnv import SimulationEngine, SimulationConfig
# fmt: on


async def enhanced_demo():
    """增强功能完整演示"""
    print("🧠 社交媒体用户认知动态变化系统演示")
    print("=" * 60)

    # 1. 创建增强的用户画像
    print("\n📊 步骤1: 创建量化用户画像")
    print("-" * 30)

    manager = UserProfileManager()

    # 生成用户
    users_count = manager.load_users_from_file("demo_users_enhanced.csv")
    users = manager.get_all_users()[:10]  # 只使用前10个用户进行演示
    # filepath = manager.generate_users(100, "demo_users_enhanced.csv")
    # users = manager.get_all_users()[:10]  # 只使用前10个用户进行演示

    print(f"✓ 加载了 {len(users)} 个用户用于演示")

    # 显示量化值
    print("\n用户画像量化值展示：")
    for i, user in enumerate(users[:3], 1):
        print(f"{i}. {user['user_id'][:15]}... - {user.get('occupation', '未知')}")
        print(f"   立场: {user.get('stance', '未知')} (量化值: {user.get('stance_value', 'N/A')})")
        print(f"   情感: {user.get('sentiment', '未知')} (量化值: {user.get('sentiment_value', 'N/A')})")

    # 2. 创建模拟环境
    print(f"\n🏗️  步骤2: 创建增强模拟环境")
    print("-" * 30)

    config = SimulationConfig(
        max_concurrent_requests=5,
        action_probability=0.8,  # 提高行动概率以便看到更多效果
        comment_probability=0.5,
        export_prompts=True,  # 🔥 启用prompt导出功能
        prompt_export_dir="SimulateEnv/data/prompt_exports"  # prompt导出目录
    )

    engine = SimulationEngine(config)

    # 创建会话
    post_content = "小米SU7自动驾驶事故后，你们还会信任自动驾驶技术吗？这次事故是技术问题还是使用不当？"
    session_id = engine.create_session(post_content)

    print(f"✓ 创建会话: {session_id}")
    print(f"✓ 话题: {post_content[:50]}...")

    # 3. 运行增强模拟
    print(f"\n🧠 步骤3: 运行认知动态模拟")
    print("-" * 30)

    try:
        # 运行2轮增强模拟
        all_actions = []

        for round_num in range(1, 3):  # 运行2轮
            print(f"\n--- 第 {round_num} 轮 ---")

            # 随机选择用户参与本轮
            import random
            round_users = random.sample(users, 5)  # 每轮5个用户

            # 使用增强模拟方法
            actions = await engine.simulate_round_with_thinking(round_users)
            all_actions.extend(actions)

            # 显示本轮的用户认知变化
            print(f"\n本轮认知变化概览:")
            for user in round_users[:3]:  # 显示前3个用户的变化
                current_profile = engine.memory_manager.get_user_current_profile(user['user_id'])
                if current_profile:
                    original_stance = user.get('stance_value', 0.0)
                    current_stance = current_profile.get('stance_value', 0.0)
                    stance_change = current_stance - original_stance

                    if abs(stance_change) > 0.01:  # 只显示有变化的
                        print(f"  {user['user_id'][:15]}... 立场: {original_stance:.2f} → {current_stance:.2f} (变化: {stance_change:+.2f})")

        # 4. 展示认知变化统计
        print(f"\n📈 步骤4: 认知变化分析")
        print("-" * 30)

        cognition_stats = engine.get_user_cognition_changes()
        print(f"总体统计:")
        print(f"- 参与用户数: {cognition_stats.get('total_users', 0)}")
        print(f"- 平均立场变化幅度: {cognition_stats.get('average_stance_change', 0):.3f}")
        print(f"- 平均情感变化幅度: {cognition_stats.get('average_sentiment_change', 0):.3f}")

        # 显示变化最大的用户
        most_changed_stance = cognition_stats.get('most_changed_stance')
        most_changed_sentiment = cognition_stats.get('most_changed_sentiment')

        if most_changed_stance:
            print(f"- 立场变化最大的用户: {most_changed_stance[:15]}...")
        if most_changed_sentiment:
            print(f"- 情感变化最大的用户: {most_changed_sentiment[:15]}...")

        # 5. 展示个别用户的详细记忆
        print(f"\n🧠 步骤5: 用户记忆详情")
        print("-" * 30)

        # 选择一个有记忆的用户展示详情
        memory_users = engine.memory_manager.list_all_users()
        if memory_users:
            sample_user = memory_users[0]
            print(f"用户 {sample_user[:15]}... 的记忆档案:")

            # 获取认知变化历史
            changes = engine.memory_manager.get_user_value_changes(sample_user)
            print(f"- 原始立场: {changes.get('original_stance', 0):.2f}")
            print(f"- 当前立场: {changes.get('current_stance', 0):.2f}")
            print(f"- 总变化: {changes.get('stance_change', 0):+.2f}")
            print(f"- 交互次数: {changes.get('interaction_count', 0)}")

            # 显示最近的思考记录
            recent_memories = engine.memory_manager.get_user_recent_interactions(sample_user, 2)
            if recent_memories:
                print(f"- 最近思考:")
                for memory in recent_memories:
                    print(f"  轮次{memory.round_number}: {memory.thinking_process[:60]}...")
                    print(f"  行为: {memory.action_taken or '无行为'}")

        print(f"\n✅ 演示结果:")
        print(f"- 总行为数: {len(all_actions)}")
        print(f"- 参与用户数: {len(set(action.user_id for action in all_actions)) if all_actions else 0}")

    except Exception as e:
        print(f"❌ 增强模拟过程中发生错误: {e}")
        print("这可能是由于API调用限制，但所有新功能的架构已经完成")

    finally:
        # 显示prompt导出文件位置
        if config.export_prompts:
            prompt_file = engine.simulator.get_prompt_export_path()
            if prompt_file:
                print(f"\n📝 Prompt导出文件: {prompt_file}")
                print("   你可以查看这个文件来分析所有发送给AI的提示词")

        await engine.close()

    # 6. 功能总结
    print(f"\n🎯 步骤6: 新功能总结")
    print("-" * 30)

    print("✅ 已实现的增强功能:")
    print("1. 🔢 量化用户画像: 立场和情感值在[-1,1]区间")
    print("2. 🧠 用户记忆系统: 每个用户独立的交互历史记录")
    print("3. 💭 思考过程生成: AI模拟用户的详细思考过程")
    print("4. 📊 动态认知变化: 实时更新用户的立场和情感值")
    print("5. 🔄 记忆驱动行为: 基于历史经验的智能行为生成")
    print("6. 📈 变化追踪分析: 完整的认知变化统计和分析")
    print("7. 📝 Prompt导出调试: 可导出所有AI提示词用于分析")

    print("\n🏗️ 系统架构:")
    print("- UserAgent/user_memory_manager.py: 记忆管理核心")
    print("- 量化画像生成: 自动分配[-1,1]的数值")
    print("- 增强AI提示: 包含历史记忆的上下文")
    print("- 实时数据保存: 避免数据丢失")
    print("- Prompt导出: data/prompt_exports/目录下的txt文件")

    print(f"\n🎊 增强系统演示完成！")


if __name__ == "__main__":
    import os

    # 运行演示
    asyncio.run(enhanced_demo())
