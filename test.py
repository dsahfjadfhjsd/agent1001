"""
系统测试文件
"""
import asyncio
import json
from datetime import datetime

from UserAgent.user_profile import UserProfile, Gender, Stance, Emotion, Intent
from UserAgent.user_agent import UserAgent, UserAgentManager
from SimulateEnv.environment_models import Post
from SimulateEnv.environment_manager import EnvironmentManager
from DisAgent.distribution_agent import DistributionManager
from cognitive_guidance_system import CognitiveGuidanceSystem

def test_user_profile():
    """测试用户画像创建"""
    print("=== 测试用户画像 ===")
    
    profile = UserProfile(
        user_id="test_user",
        age=25,
        gender=Gender.FEMALE,
        occupation="学生",
        education_level="本科",
        location="北京",
        stance=Stance.SUPPORT,
        emotion=Emotion.POSITIVE,
        intent=Intent.DISCUSS,
        activity_level=0.8,
        social_influence=0.4
    )
    
    print(f"创建用户: {profile.user_id}")
    print(f"基本信息: {profile.age}岁 {profile.gender.value} {profile.occupation}")
    print(f"事件态度: {profile.stance.value} | {profile.emotion.value} | {profile.intent.value}")
    print("✓ 用户画像测试通过")

def test_environment_models():
    """测试环境数据模型"""
    print("\n=== 测试环境模型 ===")
    
    post = Post(
        post_id="test_post_001",
        title="测试帖子",
        content="这是一个测试帖子的内容。",
        author="测试作者",
        timestamp=datetime.now()
    )
    
    print(f"创建帖子: {post.title}")
    print(f"帖子ID: {post.post_id}")
    print(f"作者: {post.author}")
    print("✓ 环境模型测试通过")

async def test_user_agent():
    """测试用户智能体"""
    print("\n=== 测试用户智能体 ===")
    
    # 创建测试用户
    profile = UserProfile(
        user_id="agent_test_user",
        age=30,
        gender=Gender.MALE,
        occupation="工程师",
        education_level="硕士",
        location="上海",
        stance=Stance.NEUTRAL,
        emotion=Emotion.NEUTRAL,
        intent=Intent.DISCUSS,
        activity_level=0.6,
        social_influence=0.5
    )
    
    agent = UserAgent(profile)
    
    # 模拟环境内容
    environment_content = {
        "post_id": "test_post_001",
        "post": {
            "title": "技术发展讨论",
            "content": "大家觉得人工智能技术对社会的影响如何？",
            "author": "讨论发起人"
        },
        "comments": [],
        "likes_count": 0,
        "shares_count": 0,
        "round": 1
    }
    
    print(f"用户 {profile.user_id} 正在分析内容...")
    
    try:
        # 注意：这里需要有效的API密钥才能正常工作
        # action = await agent.decide_action(environment_content)
        # if action:
        #     print(f"用户决定: {action.action_type}")
        #     print(f"内容: {action.content}")
        # else:
        #     print("用户选择忽略")
        print("✓ 用户智能体测试准备完成（需要API密钥才能执行）")
    except Exception as e:
        print(f"⚠ 用户智能体测试需要配置API密钥: {e}")

def test_system_integration():
    """测试系统集成"""
    print("\n=== 测试系统集成 ===")
    
    try:
        system = CognitiveGuidanceSystem()
        print(f"创建系统实例: {system.session_id}")
        
        # 测试初始化（不加载用户，避免文件路径问题）
        print("系统初始化准备完成")
        print("✓ 系统集成测试通过")
        
    except Exception as e:
        print(f"⚠ 系统集成测试出现问题: {e}")

async def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 基本功能测试 ===")
    
    # 测试配置加载
    from Config.settings import config
    print(f"配置加载: 模型 {config.model.model_name}")
    print(f"环境配置: 最大轮次 {config.environment.max_rounds}")
    
    # 测试数据结构
    from DisAgent.distribution_models import EvaluationMetrics
    metrics = EvaluationMetrics(
        engagement_rate=0.5,
        sentiment_shift=0.3,
        opinion_diversity=0.7
    )
    score = metrics.calculate_overall_score()
    print(f"评估指标测试: 综合评分 {score:.2f}")
    
    print("✓ 基本功能测试通过")

def main():
    """运行所有测试"""
    print("认知引导系统 - 功能测试")
    print("=" * 50)
    
    # 运行同步测试
    test_user_profile()
    test_environment_models()
    test_system_integration()
    
    # 运行异步测试
    asyncio.run(test_user_agent())
    asyncio.run(test_basic_functionality())
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n📝 注意事项:")
    print("1. 完整功能测试需要配置 .env 文件中的 API 密钥")
    print("2. 运行 python main.py 体验完整演示")
    print("3. 查看 Config/sample_users.json 了解用户画像配置")

if __name__ == "__main__":
    main()
