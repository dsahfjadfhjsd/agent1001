"""
认知引导系统主入口
"""
import asyncio
import json
from cognitive_guidance_system import CognitiveGuidanceSystem, quick_simulation


async def demo_simulation():
    """演示模拟"""
    print("=== 认知引导系统演示 ===")
    print("正在初始化系统...")

    # 创建系统实例
    system = CognitiveGuidanceSystem()
    system.initialize_system()

    # 准备测试内容
    post_data = {
        "title": "关于远程工作的讨论",
        "content": "随着科技发展，远程工作变得越来越普遍。你认为远程工作对工作效率和生活质量有什么影响？欢迎分享你的观点和经验。",
        "author": "主持人",
        "tags": ["工作", "远程", "效率", "生活"]
    }

    print(f"开始模拟帖子: {post_data['title']}")

    # 执行模拟
    try:
        result = await system.start_simulation(post_data, max_rounds=3)

        # 显示结果摘要
        print("\n=== 模拟结果摘要 ===")
        print(f"会话ID: {result['session_id']}")
        print(f"参与轮次: {len(result['simulation_rounds'])}")
        print(f"目标用户数: {len(result['distribution_result']['target_users'])}")

        metrics = result['final_metrics']
        print(f"最终参与率: {metrics['engagement_rate']:.2%}")
        print(f"情感影响度: {metrics['sentiment_shift']:.2f}")
        print(f"观点多样性: {metrics['opinion_diversity']:.2f}")
        print(f"触达人数: {metrics['reach']}")

        # 显示环境数据摘要
        env_data = result['environment_data']
        print(f"\n=== 交互统计 ===")
        print(f"总点赞数: {env_data['stats']['likes']}")
        print(f"总评论数: {env_data['stats']['comments']}")
        print(f"总分享数: {env_data['stats']['shares']}")

        print(f"\n详细报告已保存，请查看 SimulateEnv/outputs/ 目录")

    except Exception as e:
        print(f"模拟过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


async def batch_experiment_demo():
    """批量实验演示"""
    print("\n=== 批量实验演示 ===")

    system = CognitiveGuidanceSystem()
    system.initialize_system()

    # 准备多个测试内容
    posts = [
        {
            "title": "环保话题讨论",
            "content": "气候变化是当今世界面临的重大挑战，我们每个人都应该采取行动。你在日常生活中采取了哪些环保措施？",
            "tags": ["环保", "气候", "可持续发展"]
        },
        {
            "title": "教育创新思考",
            "content": "在线教育正在改变传统的学习方式。你认为在线教育和传统教育各有什么优缺点？",
            "tags": ["教育", "在线学习", "创新"]
        }
    ]

    try:
        result = await system.run_batch_experiments(posts, max_rounds=2)

        print(f"完成 {result['analysis']['total_experiments']} 个实验")
        print(f"平均参与率: {result['analysis']['average_metrics'].get('engagement_rate', 0):.2%}")
        print(f"最佳实验: {result['analysis']['best_experiment_id']}")

    except Exception as e:
        print(f"批量实验中出现错误: {e}")


def main():
    """主函数"""
    print("认知引导系统 v1.0")
    print("选择演示模式:")
    print("1. 单次模拟演示")
    print("2. 批量实验演示")
    print("3. 快速模拟")

    choice = input("请输入选择 (1-3): ").strip()

    if choice == "1":
        asyncio.run(demo_simulation())
    elif choice == "2":
        asyncio.run(batch_experiment_demo())
    elif choice == "3":
        content = input("请输入要模拟的帖子内容: ").strip()
        if content:
            async def run_quick():
                result = await quick_simulation(content, max_rounds=2)
                print(f"模拟完成！参与率: {result['final_metrics']['engagement_rate']:.2%}")
            asyncio.run(run_quick())
    else:
        print("无效选择，退出程序")


if __name__ == "__main__":
    main()
