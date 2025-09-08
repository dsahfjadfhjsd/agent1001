# -*- coding: utf-8 -*-
"""
Super Demo (demo_2.py)

Multi-scenario distribution + EvalAgent-driven intelligent optimization using DISTAgent (DisAgent) and EvalAgent.

Pipeline:
1) Load posts and initialize users
2) Build dynamic multi-scenarios (awareness, discussion, conversion) based on content
3) For each scenario, use DISTAgent to generate strategy (if available) and simulate with SimulationEngine
4) Evaluate with EvalAgent to get multi-dimensional metrics and optimization feedback
5) Apply prompt/strategy/parameter updates back to DISTAgent and run a post-optimization scenario pass

Outputs:
- Output/exports/{batch_id}/simulation_results.json
- Output/exports/{batch_id}/evaluation_results.json
- Prompt exports under Output/prompt_exports/{batch_id}/...

Requirements:
- Config/.env should include OPENAI_API_KEY and optionally OPENAI_BASE_URL for full DISTAgent cognitive features
- See requirements.txt for dependencies
"""

import asyncio
import json
import os
import random
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
from dotenv import load_dotenv
import sys

# Reduce noisy warnings (e.g., from torchvision)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

# Ensure env
load_dotenv(dotenv_path=os.path.join("Config", ".env"))

# Fix Windows Proactor warnings by using Selector policy
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Project modules
from UserAgent.user_profile_manager import UserProfileManager
from SimulateEnv import SimulationEngine, SimulationConfig

# DisAgent (DISTAgent)
from DisAgent.distagent_framework import (
    DISTAgent,
    DISTAgentConfig,
    ContentDistributionTask,
    create_distagent,
)

# EvalAgent (5-module evaluation system)
from EvalAgent import create_eval_agent, EvalAgent


@dataclass
class Scenario:
    scenario_id: str
    name: str
    description: str
    cognitive_objectives: Dict[str, Any]
    platform_preferences: List[str]
    keywords: List[str]


class MultiScenarioSmartDemo:
    """Multi-scenario distribution + EvalAgent optimization using DISTAgent and SimulationEngine."""

    def __init__(self, batch_id: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_id = batch_id or f"super_demo_{timestamp}"
        self.out_dir = Path(f"Output/exports/{self.batch_id}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.posts_data: List[Dict[str, Any]] = []
        self.users_data: List[Dict[str, Any]] = []
        self.scenarios: List[Scenario] = []

        # Runtime knobs (adaptive via optimization)
        self.posts_per_scenario: int = 2
        self.users_per_post: int = 6
        self.rounds_per_post: int = 1

        # Engine
        self.sim_engine: Optional[SimulationEngine] = None

        # Results structure (EvalAgent-friendly)
        self.simulation_results: Dict[str, Any] = {
            "batch_id": self.batch_id,
            "start_time": datetime.now().isoformat(),
            "rounds": [],
            # Flattened posts (EvalAgent's DataCollectionModule reads top-level posts)
            "posts": [],
        }

        # Real-time UI callback: callable(update_type: str, data: Dict[str, Any]) -> None
        self.realtime_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

        print(f"🚀 超级多场景分发 + 智能优化演示初始化，批次: {self.batch_id}")

        # Create DISTAgent with robust defaults
        dist_config = {
            "cognitive_config": {
                "model_name": os.getenv("OPENAI_MODEL", "qwen-max"),
                "num_workers": 1,
                "temperature": 0.3,
                "max_tokens": 1024,
                "enable_differential_privacy": True,
            },
            "memory_config": {
                "enable_hierarchical_retrieval": True,
                "embedding_dim": 768,
                "memory_decay_factor": 0.95,
            },
            "tool_config": {
                "enable_roberta_sentiment": True,
                "enable_propagation_analysis": True,
                "enable_batch_processing": True,
            },
            "action_config": {
                "enable_mcts": True,
                "mcts_simulations": 80,
                "mcts_exploration_constant": 1.4,
                "enable_reinforcement_learning": True,
                # Do not split platforms across strategies
                "single_platform_mode": True,
                "default_platform": "weibo",
            },
            "evaluation_config": {
                "enable_multi_dimensional_feedback": True,
            },
            "global_config": {
                "agent_id": f"dist_{self.batch_id}",
                "max_concurrent_tasks": 4,
            },
            "eval_agent_config": {
                # Not used directly here, we instantiate EvalAgent separately
            },
        }

        # Instantiate agents
        self.dist_agent: Optional[DISTAgent] = None
        self.eval_agent: Optional[EvalAgent] = None

        try:
            self.dist_agent = create_distagent(dist_config)
        except Exception as e:
            print(f"⚠️ 创建DISTAgent失败，将使用降级模式: {e}")
            self.dist_agent = None

        try:
            self.eval_agent = create_eval_agent(
                {
                    "data_collection": {
                        "enable_real_time": True,
                        "privacy_protection": True,
                        "privacy_epsilon": 1.0,
                    },
                    "effect_analysis": {
                        "enable_deep_engagement": True,
                        "propagation_analysis": True,
                        "enable_distributed": False,
                    },
                    "cognitive_assessment": {
                        "consistency_threshold": 0.8,
                        "sentiment_analysis": True,
                        "privacy_protection": True,
                    },
                    "optimization_feedback": {
                        "enable_auto_optimization": True,
                        "learning_rate": 0.02,
                        "prompt_learning": True,
                        "adaptive_weighting": True,
                    },
                }
            )
        except Exception as e:
            print(f"❌ 创建EvalAgent失败: {e}")
            raise

    async def initialize(self):
        """Initialize DISTAgent modules and simulation engine."""
        # Initialize DISTAgent if available
        if self.dist_agent is not None:
            try:
                await self.dist_agent.initialize()
                print("✅ DISTAgent 初始化成功 (全功能模式)")
            except Exception as e:
                print(f"⚠️ DISTAgent 初始化失败，启用降级模式: {e}")
                self.dist_agent = None

        # Simulation engine with prompt export
        self.sim_engine = SimulationEngine(
            SimulationConfig(
                max_concurrent_requests=4,
                action_probability=0.7,
                comment_probability=0.5,
                export_prompts=True,
                prompt_export_dir=f"Output/prompt_exports/{self.batch_id}",
            )
        )

    # --------------------- Data Loading ---------------------
    def load_posts_from_csv(
        self, csv_path: str = "Data/integrated_data/XMSU7D_integrated_articles.csv", max_posts: int = 12
    ) -> List[Dict[str, Any]]:
        print(f"📂 从 {csv_path} 加载帖子...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠️ 无法读取CSV，使用示例数据: {e}")
            return self._load_sample_posts()

        if len(df) > max_posts:
            df = df.sample(n=max_posts, random_state=42)

        posts: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            pid = str(row.get("article_id", row.get("post_id", f"post_{len(posts)}")))
            content = str(row.get("content", row.get("text", "")))
            posts.append(
                {
                    "post_id": pid,
                    "content": content,
                    "title": str(row.get("title", "")),
                    "platform": str(row.get("platform", "unknown")),
                    "original_likes": int(row.get("like_count", 0)) if "like_count" in row else 0,
                    "original_comments": int(row.get("comment_count", 0)) if "comment_count" in row else 0,
                }
            )
        print(f"✅ 加载 {len(posts)} 个帖子")
        return posts

    def _load_sample_posts(self) -> List[Dict[str, Any]]:
        return [
            {
                "post_id": "sample_1",
                "content": "人工智能正在改变我们的生活方式，从智能手机到自动驾驶，AI技术无处不在。",
                "title": "AI技术的发展与应用",
                "platform": "social",
                "original_likes": 128,
                "original_comments": 45,
            },
            {
                "post_id": "sample_2",
                "content": "气候变化是当今世界面临的重大挑战，需要全球共同努力应对。",
                "title": "气候变化与环境保护",
                "platform": "social",
                "original_likes": 89,
                "original_comments": 32,
            },
            {
                "post_id": "sample_3",
                "content": "教育是社会进步的基石，在线教育为更多人提供了学习机会。",
                "title": "在线教育的发展前景",
                "platform": "social",
                "original_likes": 156,
                "original_comments": 67,
            },
            {
                "post_id": "sample_4",
                "content": "健康的生活方式包括合理饮食、适量运动和充足睡眠。",
                "title": "健康生活的重要性",
                "platform": "social",
                "original_likes": 203,
                "original_comments": 78,
            },
            {
                "post_id": "sample_5",
                "content": "科技创新推动经济发展，新兴产业成为增长新动力。",
                "title": "科技创新与经济发展",
                "platform": "social",
                "original_likes": 174,
                "original_comments": 54,
            },
        ]

    def initialize_users(self, total_users: int = 24):
        print("👥 初始化用户画像...")
        upm = UserProfileManager()
        try:
            upm.generate_users(count=total_users, filename=f"super_demo_users_{self.batch_id}.csv")
        except Exception:
            pass
        users = upm.get_all_users()
        self.users_data = users[:total_users] if len(users) > total_users else users
        print(f"✅ 用户数: {len(self.users_data)}")

    # --------------------- Scenario Planning ---------------------
    def build_dynamic_scenarios(self, posts: List[Dict[str, Any]]) -> List[Scenario]:
        """Build 3 dynamic scenarios by content theme and objectives."""
        print("🧭 生成多场景分发计划 (基于内容主题和目标)")
        scenarios: List[Scenario] = []

        scenarios.append(
            Scenario(
                scenario_id="awareness",
                name="品牌认知/科普扩散",
                description="扩大覆盖与话题曝光，提升用户基本认知",
                cognitive_objectives={
                    "stance": "neutral",
                    "emotion": "curiosity",
                    "intent": "inform",
                },
                platform_preferences=["weibo", "twitter", "wechat"],
                keywords=["AI", "人工智能", "科技", "创新"],
            )
        )
        scenarios.append(
            Scenario(
                scenario_id="discussion",
                name="理性讨论/观点碰撞",
                description="促进理性高质量讨论，提升思考深度与认知一致性",
                cognitive_objectives={
                    "stance": "moderate",
                    "emotion": "calm",
                    "intent": "discuss",
                },
                platform_preferences=["zhihu", "weibo"],
                keywords=["教育", "社会", "环境", "气候"],
            )
        )
        scenarios.append(
            Scenario(
                scenario_id="conversion",
                name="行动转化/关注订阅",
                description="引导用户关注、转发或报名，提升转化率",
                cognitive_objectives={
                    "stance": "positive",
                    "emotion": "motivational",
                    "intent": "call_to_action",
                },
                platform_preferences=["wechat", "weibo"],
                keywords=["健康", "希望", "建议", "重要性"],
            )
        )

        # Persist
        self.scenarios = scenarios
        print(f"✅ 场景数: {len(scenarios)} -> {[s.name for s in scenarios]}")
        return scenarios

    def _select_posts_for_scenario(self, scenario: Scenario, pool: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        # Keyword match first, else fallback to random
        matched = [p for p in pool if any(kw in p.get("content", "") for kw in scenario.keywords)]
        if len(matched) < k:
            remaining = [p for p in pool if p not in matched]
            random.shuffle(remaining)
            matched += remaining[: max(0, k - len(matched))]
        return matched[:k]

    # --------------------- ToolModule-Aware Selection Helpers ---------------------
    def _map_objectives_to_user_filters(self, scenario: Scenario) -> Dict[str, str]:
        """Map scenario cognitive objectives to user profile filters (Chinese labels).

        Returns a dict like {"stance": "中立"|"支持"|"反对", "sentiment": "积极"|"中立"|"消极"}
        """
        stance_map = {
            "moderate": "中立",
            "neutral": "中立",
            "positive": "支持",
            "negative": "反对",
        }
        sentiment_map = {
            "positive": "积极",
            "neutral": "中立",
            "negative": "消极",
            # map some common intents/emotions to closest sentiment bucket
            "calm": "中立",
            "curiosity": "中立",
            "motivational": "积极",
        }

        obj = scenario.cognitive_objectives or {}
        stance_cn = stance_map.get(str(obj.get("stance", "")).lower(), None)
        # try sentiment first, then emotion as fallback
        senti_key = str(obj.get("sentiment", obj.get("emotion", ""))).lower()
        sentiment_cn = sentiment_map.get(senti_key, None)
        return {"stance": stance_cn, "sentiment": sentiment_cn}

    # --------------------- Realtime Emission Helper ---------------------
    def _emit(self, update_type: str, data: Dict[str, Any]):
        """Emit real-time updates if a callback is registered.

        update_type examples: 'post_distributed' | 'user_simulation_start' | 'user_action' | 'simulation_progress'
        """
        try:
            if callable(self.realtime_callback):
                self.realtime_callback(update_type, data)
        except Exception:
            # Ignore UI errors
            pass

    async def _rank_users_by_propagation(self, user_ids: List[str], post: Dict[str, Any]) -> List[str]:
        """Rank users by propagation influence using ToolModule.propagation analysis.
        Fallback to original order if DISTAgent/ToolModule not available.
        """
        if not self.dist_agent or not getattr(self.dist_agent, "tool_module", None):
            return user_ids

        try:
            pr = await self.dist_agent.tool_module.analyze_propagation(
                user_ids=user_ids,
                content_metadata={"post_id": post.get("post_id"), "posts": [post]},
            )
            # Build influence map from propagation_paths
            paths = pr.get("propagation_paths", []) or []
            inf_map = {str(n.get("node_id")): float(n.get("influence", 0.0)) for n in paths}
            # Boost by user activity_level if available
            activity_boost = {"high": 1.15, "medium": 1.05, "low": 1.0}
            profile_map = {str(u.get("user_id", "")): u for u in self.users_data}
            def score(uid: str) -> float:
                base = inf_map.get(uid, 0.0)
                act = str(profile_map.get(uid, {}).get("activity_level", "low")).lower()
                return base * activity_boost.get(act, 1.0)
            ranked = sorted(user_ids, key=score, reverse=True)
            return ranked
        except Exception:
            return user_ids

    def _filter_users_by_cognition(self, user_ids: List[str], filters: Dict[str, Optional[str]]) -> List[str]:
        """Filter users by stance/sentiment labels if provided."""
        if not user_ids:
            return []
        profile_map = {str(u.get("user_id", "")): u for u in self.users_data}
        target_stance = filters.get("stance")
        target_sentiment = filters.get("sentiment")

        def ok(uid: str) -> bool:
            u = profile_map.get(uid, {})
            if target_stance and u.get("stance") not in (target_stance, None, ""):
                return False
            if target_sentiment and u.get("sentiment") not in (target_sentiment, None, ""):
                return False
            return True

        return [uid for uid in user_ids if ok(uid)]

    # --------------------- Simulation & Distribution ---------------------
    async def run_scenario(self, scenario: Scenario, round_index: int, available_posts: List[Dict[str, Any]]):
        assert self.sim_engine is not None
        print(f"\n🎯 执行场景 {round_index}: {scenario.name}")

        # Choose scenario posts
        selected_posts = self._select_posts_for_scenario(scenario, available_posts, self.posts_per_scenario)
        selected_post_ids = [p["post_id"] for p in selected_posts]
        print(f"   选中帖子: {selected_post_ids}")

        # Prepare target users
        all_user_ids = [str(u.get("user_id", f"u_{i}")) for i, u in enumerate(self.users_data)]
        target_user_ids = all_user_ids[: max(self.users_per_post * self.posts_per_scenario, 1)]

        # If DISTAgent available, ask it to plan
        action_strategy: Dict[str, Any] = {}
        user_assignments: Dict[str, List[str]] = {}
        platform_strategy: Dict[str, List[str]] = {}
        if self.dist_agent is not None:
            try:
                task = ContentDistributionTask(
                    task_id=f"scenario_{scenario.scenario_id}_{round_index}",
                    content_data={
                        "type": "multi_post_distribution",
                        "scenario": scenario.scenario_id,
                        "posts": selected_posts,
                        "round": round_index,
                    },
                    target_users=target_user_ids,
                    distribution_params={
                        "cognitive_objectives": scenario.cognitive_objectives,
                        "platform_preferences": scenario.platform_preferences,
                        "posts_per_round": self.posts_per_scenario,
                        "users_per_post": self.users_per_post,
                        "priority": "high" if round_index == 1 else "medium",
                    },
                    priority="high" if round_index == 1 else "medium",
                    created_at=datetime.now(),
                )
                plan = await self.dist_agent.process_task(task)
                action = plan.get("action_strategy", {})
                action_strategy = action.get("optimal_strategy") or action.get("strategy", {})
                user_assignments = action.get("user_assignments", {})
                platform_strategy = action.get("platform_selection", {}) or action.get("platform_strategy", {})
                print("   🧠 DISTAgent策略已生成")
                # 展示策略生成原因（便于定位无策略的原因）
                try:
                    gen_reason = action.get("generation_reason", {}) or {}
                    # 仅在非OK时打印诊断，避免与后续“分发原因”重复干扰
                    if gen_reason.get('reason_code') and gen_reason.get('reason_code') != 'ok':
                        print(
                            f"   策略生成诊断 → code={gen_reason.get('reason_code')}, "
                            f"message={gen_reason.get('reason_message')}, "
                            f"contents={gen_reason.get('available_contents_count')}, users={gen_reason.get('available_users_count')}"
                        )
                except Exception:
                    pass
                # 展示分发策略详情（不改模拟模块，仅打印）
                try:
                    selected = action_strategy.get("selected_contents", [])
                    sentiment = action_strategy.get("sentiment")
                    print(f"   分发策略详情 → selected_contents={selected}, timing=immediate, sentiment={sentiment}")
                    # 贴出所选帖子的正文与用户分配
                    post_map = {str(p.get("post_id")): p for p in selected_posts}
                    for pid in selected:
                        pid_str = str(pid)
                        p = post_map.get(pid_str, {})
                        content_text = p.get("content", "")
                        print(f"     · 帖子 {pid_str} 正文：{content_text}")
                        assigned = user_assignments.get(pid_str, [])
                        print(f"       用户分配({len(assigned)}): {assigned}")
                except Exception:
                    pass
            except Exception as e:
                print(f"   ⚠️ DISTAgent策略生成失败，使用降级策略: {e}")
        else:
            print("   ⚠️ DISTAgent不可用，使用随机分配策略")

        # Simulate each post
        round_results = {
            "round_number": round_index,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "posts": [],
            "total_actions": 0,
            "start_time": datetime.now().isoformat(),
        }

        total_posts_in_round = len(selected_posts) if selected_posts else 0
        for idx, post in enumerate(selected_posts, 1):
            post_id = post["post_id"]
            # Map assignments to user profiles
            assigned_ids = user_assignments.get(post_id) or target_user_ids[: self.users_per_post]

            # Integrate ToolModule results: filter by cognition and rank by propagation
            filters = self._map_objectives_to_user_filters(scenario)
            filtered = self._filter_users_by_cognition(assigned_ids, filters)
            # If filter too strict, fall back to original assigned ids
            base_candidates = filtered if len(filtered) >= max(1, self.users_per_post // 2) else assigned_ids
            ranked_ids = await self._rank_users_by_propagation(base_candidates, post)
            final_ids = (ranked_ids + [uid for uid in assigned_ids if uid not in ranked_ids])[: self.users_per_post]
            # 展示生效后的策略（最终用于模拟的用户列表）
            try:
                print(f"   ✅ 生效策略 → 帖子 {post_id} 最终用户({len(final_ids)}): {final_ids}")
                # 组合并展示分发原因（场景×情感×过滤×排序×人数）
                try:
                    scenario_label = f"{scenario.name}({scenario.scenario_id})"
                    sent = (sentiment_summary or {}).get("sentiment")
                    filtering_info = f"认知过滤保留 {len(filtered)}/{len(assigned_ids)}"
                    ranking_info = "传播潜力排序优先"
                    selection_mode = "智能策略(MCTS)" if round_index > 1 else "初轮探索/多样性随机"
                    parts = [
                        f"场景: {scenario_label}",
                        f"情感: {sent}" if sent else None,
                        f"选择模式: {selection_mode}",
                        filtering_info,
                        ranking_info,
                        f"最终分配: {len(final_ids)} 人"
                    ]
                    reason_text = "；".join([p for p in parts if p])
                    print(f"   📌 分发原因 → 帖子 {post_id}: {reason_text}")
                except Exception:
                    pass
            except Exception:
                pass

            # Build profile list in same order
            profile_map = {str(u.get("user_id", "")): u for u in self.users_data}
            selected_users = [profile_map[uid] for uid in final_ids if uid in profile_map]
            if not selected_users:
                selected_users = self.users_data[: self.users_per_post]

            # Generate content variant using sentiment analysis and scenario intent
            variant_content = post.get("content", "")
            sentiment_summary = {}
            if self.dist_agent and getattr(self.dist_agent, "tool_module", None):
                try:
                    sres = await self.dist_agent.tool_module.analyze_sentiment(variant_content, context={"post_id": post_id})
                    sentiment_summary = sres or {}
                    tag = {
                        "awareness": "【科普扩散】",
                        "discussion": "【理性讨论】",
                        "conversion": "【行动建议】",
                    }.get(scenario.scenario_id, "【主题】")
                    # Light-touch varianting to reflect scenario intent and sentiment state
                    if (sres.get("sentiment") == "negative") and scenario.scenario_id in ("discussion", "conversion"):
                        variant_content = f"{tag}欢迎提供建设性意见：" + variant_content
                    else:
                        variant_content = f"{tag}" + variant_content
                except Exception:
                    pass

            # Realtime: announce post distribution to UI
            try:
                self._emit(
                    "post_distributed",
                    {
                        "post_id": post_id,
                        "title": post.get("title", f"帖子 {post_id}"),
                        "content": variant_content,
                        "round_type": "optimized" if round_index > len(self.scenarios) else "initial",
                        "scenario_name": scenario.name,
                        "scenario_id": scenario.scenario_id,
                    },
                )
            except Exception:
                pass

            self.sim_engine.create_session(post_content=variant_content, post_id=post_id, batch_id=self.batch_id)
            # 如果是“加载已存在的会话”，在这里展示会话帖子的内容（不改模拟模块，仅打印）
            try:
                summary = self.sim_engine.get_session_summary(post_id)
                if summary:
                    src_batch = summary.get("batch_id")
                    total_rounds = int(summary.get("total_rounds", 0)) if str(summary.get("total_rounds", "")).isdigit() else summary.get("total_rounds", 0)
                    loaded_from_history = (src_batch and src_batch != self.batch_id) or (total_rounds and total_rounds > 0)
                    if loaded_from_history:
                        print(f"   🔄 加载已存在的会话: {post_id}")
                        if src_batch:
                            print(f"      来源批次: {src_batch}")
                        print(f"      帖子内容：{summary.get('post_content', '')}")
                        print(f"      已有轮次: {summary.get('total_rounds')}, 已有行为数: {summary.get('total_actions')}, 已有评论数: {summary.get('total_comments')}")
            except Exception:
                pass

            post_actions: List[Any] = []
            # Realtime: announce user thinking start
            try:
                for u in selected_users:
                    uid = str(u.get("user_id", ""))
                    if uid:
                        self._emit("user_simulation_start", {"user_id": uid, "post_id": post_id})
            except Exception:
                pass
            for inner in range(1, self.rounds_per_post + 1):
                inner_actions = await self.sim_engine.simulate_round_with_thinking(selected_users)
                if inner_actions:
                    post_actions.extend(inner_actions)

            # Aggregate comments and likes
            comments = []
            likes = 0
            for a in post_actions:
                try:
                    at = getattr(a, "action_type").value  # enum -> str
                except Exception:
                    at = getattr(a, "action_type", "")
                if at in ["comment_post", "comment"]:
                    comments.append(
                        {
                            "author": getattr(a, "user_id", "Unknown"),
                            "content": getattr(a, "content", ""),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                elif at == "like_post":
                    likes += 1

            action_types: Dict[str, int] = {}
            for a in post_actions:
                try:
                    at = getattr(a, "action_type").value
                except Exception:
                    at = getattr(a, "action_type", "")
                action_types[at] = action_types.get(at, 0) + 1

            # Realtime: emit each user action to UI
            def _map_action_type(at_str: str) -> str:
                m = {
                    "like_post": "like",
                    "comment_post": "comment",
                    "share_post": "share",
                    "like_comment": "like",
                    "comment_comment": "comment",
                }
                return m.get(at_str, at_str or "action")

            try:
                for a in post_actions:
                    try:
                        at = getattr(a, "action_type").value
                    except Exception:
                        at = getattr(a, "action_type", "")
                    self._emit(
                        "user_action",
                        {
                            "user_id": getattr(a, "user_id", "Unknown"),
                            "post_id": post_id,
                            "action_type": _map_action_type(at),
                            "content": getattr(a, "content", None),
                        },
                    )
            except Exception:
                pass

            post_result = {
                "post_id": post_id,
                "title": post.get("title", f"帖子 {post_id}"),
                "content": variant_content,
                "users_count": len(selected_users),
                "actions_count": len(post_actions),
                "action_types": action_types,
                "comments": comments,
                "likes": likes,
                "selected_user_ids": final_ids if 'final_ids' in locals() else assigned_ids,
                "analysis": {
                    "sentiment": sentiment_summary,
                },
            }
            round_results["posts"].append(post_result)
            round_results["total_actions"] += len(post_actions)

            # Realtime: progress update within round
            try:
                if total_posts_in_round:
                    self._emit(
                        "simulation_progress",
                        {
                            "round_index": round_index,
                            "processed_posts": idx,
                            "total_posts": total_posts_in_round,
                        },
                    )
            except Exception:
                pass

        round_results["end_time"] = datetime.now().isoformat()
        round_results["duration_minutes"] = (
            datetime.fromisoformat(round_results["end_time"]) - datetime.fromisoformat(round_results["start_time"])
        ).total_seconds() / 60.0

        # Persist round
        self.simulation_results["rounds"].append(round_results)
        # Also flatten posts for EvalAgent convenience
        self.simulation_results["posts"].extend(round_results["posts"])
        self._save_json(self.out_dir / "simulation_results.json", self.simulation_results)

        print(
            f"   ✅ 场景完成: 帖子 {len(round_results['posts'])} | 行为 {round_results['total_actions']} | 时长 {round_results['duration_minutes']:.2f} 分钟"
        )

    # --------------------- Evaluation & Optimization ---------------------
    async def evaluate_and_optimize(self) -> Dict[str, Any]:
        assert self.eval_agent is not None
        print("\n🧪 使用 EvalAgent 进行深度评估与优化反馈...")
        try:
            evaluation = await self.eval_agent.evaluate_distribution_performance(self.simulation_results)
            self._save_json(self.out_dir / "evaluation_results.json", evaluation)
            print("✅ 评估完成")
        except Exception as e:
            print(f"❌ EvalAgent评估失败: {e}")
            raise

        # Apply optimization back to DISTAgent
        if self.dist_agent is not None and evaluation and "optimization" in evaluation:
            opt = evaluation["optimization"]
            try:
                # Strategy weights -> ActionModule
                if "strategy_adjustments" in opt and self.dist_agent.action_module:
                    # update_strategy_weights is synchronous in current implementation
                    self.dist_agent.action_module.update_strategy_weights(opt)

                # Prompt optimization -> CognitiveFoundationModel
                prompt_opt = opt.get("prompt_optimization", {})
                if prompt_opt and self.dist_agent.cognitive_foundation and "distribution_prompt" in prompt_opt:
                    await self.dist_agent.cognitive_foundation.update_system_prompt(prompt_opt["distribution_prompt"])

                print("🔧 已将优化反馈应用到 DISTAgent")
            except Exception as e:
                print(f"⚠️ 应用优化反馈失败: {e}")

        # Adjust local knobs based on parameter tuning
        if evaluation and "optimization" in evaluation:
            tuning = evaluation["optimization"].get("parameter_tuning", {})
            self.posts_per_scenario = int(tuning.get("posts_per_round", self.posts_per_scenario))
            self.users_per_post = int(tuning.get("users_per_post", self.users_per_post))
            print(
                f"📊 更新参数: posts_per_scenario={self.posts_per_scenario}, users_per_post={self.users_per_post}"
            )

        return evaluation

    # --------------------- Orchestration ---------------------
    async def run(self, pre_eval_cycles: int = 1, post_eval_cycles: int = 1):
        print("\n🚦 开始执行多场景分发 + 智能优化管道")
        # Build scenario list once
        self.build_dynamic_scenarios(self.posts_data)

        # Track used posts
        used_ids: set = set()

        # Pre-evaluation cycles
        for cycle in range(1, pre_eval_cycles + 1):
            print(f"\n========== 评估前周期 {cycle}/{pre_eval_cycles} ==========")
            for s_idx, scenario in enumerate(self.scenarios, 1):
                available_posts = [p for p in self.posts_data if p["post_id"] not in used_ids]
                if not available_posts:
                    available_posts = self.posts_data
                current_round_index = len(self.simulation_results["rounds"]) + 1
                # Realtime: round_start for UI
                try:
                    self._emit("round_start", {"round": current_round_index, "type": "initial"})
                except Exception:
                    pass
                await self.run_scenario(scenario, round_index=current_round_index, available_posts=available_posts)
                # Realtime: round_complete for UI with cumulative stats
                try:
                    rounds = self.simulation_results.get("rounds", [])
                    stats = {
                        "total_rounds": len(rounds),
                        "total_posts": sum(len(r.get("posts", [])) for r in rounds),
                        "total_users": len(self.users_data or []),
                        "total_actions": sum(int(r.get("total_actions", 0)) for r in rounds),
                    }
                    self._emit("round_complete", {"round": current_round_index, "type": "initial", "stats": stats})
                except Exception:
                    pass
                # Mark used posts
                used_ids.update([p["post_id"] for p in self.simulation_results["rounds"][-1]["posts"]])

        # Mark boundary between pre and post rounds
        pre_rounds_count = len(self.simulation_results["rounds"])

        # Evaluate & optimize
        try:
            self._emit("evaluation_start", {"message": "Starting evaluation and optimization..."})
        except Exception:
            pass
        evaluation = await self.evaluate_and_optimize()
        try:
            self._emit("evaluation_complete", {"evaluation": evaluation})
        except Exception:
            pass

        # Post-evaluation cycles (with tuned parameters)
        for cycle in range(1, post_eval_cycles + 1):
            print(f"\n========== 评估后周期 {cycle}/{post_eval_cycles} ==========")
            for s_idx, scenario in enumerate(self.scenarios, 1):
                available_posts = [p for p in self.posts_data if p["post_id"] not in used_ids]
                if not available_posts:
                    available_posts = self.posts_data
                current_round_index = len(self.simulation_results["rounds"]) + 1
                # Realtime: round_start for optimized rounds
                try:
                    self._emit("round_start", {"round": current_round_index, "type": "optimized"})
                except Exception:
                    pass
                await self.run_scenario(scenario, round_index=current_round_index, available_posts=available_posts)
                # Realtime: round_complete for UI with cumulative stats
                try:
                    rounds = self.simulation_results.get("rounds", [])
                    stats = {
                        "total_rounds": len(rounds),
                        "total_posts": sum(len(r.get("posts", [])) for r in rounds),
                        "total_users": len(self.users_data or []),
                        "total_actions": sum(int(r.get("total_actions", 0)) for r in rounds),
                    }
                    self._emit("round_complete", {"round": current_round_index, "type": "optimized", "stats": stats})
                except Exception:
                    pass
                used_ids.update([p["post_id"] for p in self.simulation_results["rounds"][-1]["posts"]])

        # Generate multi-scenario comparison report
        report_path = self.generate_multi_scenario_comparison_report(pre_rounds_count, evaluation)
        print(f"   多场景对比报告: {report_path}")

        # Final summary
        self._print_summary()

    # --------------------- Utilities ---------------------
    def _save_json(self, path: Path, data: Any):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ 保存JSON失败 {path}: {e}")

    def _print_summary(self):
        rounds = self.simulation_results.get("rounds", [])
        stats = {
            "total_rounds": len(rounds),
            "total_posts_used": sum(len(r.get("posts", [])) for r in rounds),
            "total_actions": sum(r.get("total_actions", 0) for r in rounds),
            "duration_minutes": sum(r.get("duration_minutes", 0.0) for r in rounds),
        }
        print("\n🏁 超级多场景闭环完成")
        print(f"   轮次数: {stats['total_rounds']}")
        print(f"   使用帖子数: {stats['total_posts_used']}")
        print(f"   总行为数: {stats['total_actions']}")
        print(f"   总耗时: {stats['duration_minutes']:.2f} 分钟")
        print("\n📁 输出:")
        print(f"   模拟结果: {self.out_dir / 'simulation_results.json'}")
        print(f"   评估结果: {self.out_dir / 'evaluation_results.json'}")

    # --------------------- Reporting ---------------------
    def _aggregate_stats_by_scenario(self, rounds_subset: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics by scenario for a subset of rounds."""
        agg: Dict[str, Dict[str, Any]] = {}
        all_key = "__ALL__"

        def _ensure(key: str):
            if key not in agg:
                agg[key] = {
                    "rounds": 0,
                    "posts": 0,
                    "total_actions": 0,
                    "likes": 0,
                    "comments": 0,
                    "action_types": {},
                }

        for rd in rounds_subset:
            sid = rd.get("scenario_id", "unknown")
            _ensure(sid)
            _ensure(all_key)

            for key in (sid, all_key):
                agg[key]["rounds"] += 1
                agg[key]["posts"] += len(rd.get("posts", []))
                agg[key]["total_actions"] += int(rd.get("total_actions", 0))

            for post in rd.get("posts", []):
                likes = int(post.get("likes", 0))
                comments = len(post.get("comments", []))
                for key in (sid, all_key):
                    agg[key]["likes"] += likes
                    agg[key]["comments"] += comments

                atypes = post.get("action_types", {}) or {}
                for a_name, cnt in atypes.items():
                    for key in (sid, all_key):
                        agg[key]["action_types"][a_name] = agg[key]["action_types"].get(a_name, 0) + int(cnt)

        # derive averages
        for key, s in agg.items():
            posts = s.get("posts", 0)
            s["avg_actions_per_post"] = (s["total_actions"] / posts) if posts else 0.0
            s["avg_comments_per_post"] = (s["comments"] / posts) if posts else 0.0
            s["avg_likes_per_post"] = (s["likes"] / posts) if posts else 0.0

        return agg

    def generate_multi_scenario_comparison_report(self, pre_rounds_count: int, evaluation: Optional[Dict[str, Any]] = None) -> str:
        """Generate a markdown report comparing pre- vs post-evaluation metrics per scenario."""
        rounds_all: List[Dict[str, Any]] = self.simulation_results.get("rounds", [])
        pre_rounds = rounds_all[:pre_rounds_count]
        post_rounds = rounds_all[pre_rounds_count:]

        pre = self._aggregate_stats_by_scenario(pre_rounds)
        post = self._aggregate_stats_by_scenario(post_rounds)

        # Collect all scenario keys (exclude __ALL__ for per-scenario table, handle ALL separately)
        scenario_keys = sorted({k for k in list(pre.keys()) + list(post.keys()) if k != "__ALL__"})

        lines: List[str] = []
        lines.append(f"# 多场景结果对比报告\n")
        lines.append(f"批次: {self.batch_id}")
        lines.append(f"生成时间: {datetime.now().isoformat()}\n")

        # Overall summary
        lines.append("## 总览（全部场景合并）")
        all_pre = pre.get("__ALL__", {})
        all_post = post.get("__ALL__", {})

        def pct_delta(new: float, old: float) -> str:
            if old == 0:
                return "N/A"
            return f"{(new - old) / old:+.1%}"

        lines.append("- 指标对比：")
        lines.append(f"  - 轮次数: 前 {all_pre.get('rounds', 0)} | 后 {all_post.get('rounds', 0)}")
        lines.append(f"  - 帖子数: 前 {all_pre.get('posts', 0)} | 后 {all_post.get('posts', 0)} ({pct_delta(all_post.get('posts', 0), all_pre.get('posts', 0))})")
        lines.append(f"  - 总行为: 前 {all_pre.get('total_actions', 0)} | 后 {all_post.get('total_actions', 0)} ({pct_delta(all_post.get('total_actions', 0), all_pre.get('total_actions', 0))})")
        lines.append(f"  - 点赞: 前 {all_pre.get('likes', 0)} | 后 {all_post.get('likes', 0)} ({pct_delta(all_post.get('likes', 0), all_pre.get('likes', 0))})")
        lines.append(f"  - 评论: 前 {all_pre.get('comments', 0)} | 后 {all_post.get('comments', 0)} ({pct_delta(all_post.get('comments', 0), all_pre.get('comments', 0))})")
        lines.append(f"  - 平均行为/帖: 前 {all_pre.get('avg_actions_per_post', 0):.2f} | 后 {all_post.get('avg_actions_per_post', 0):.2f} ({pct_delta(all_post.get('avg_actions_per_post', 0.0), all_pre.get('avg_actions_per_post', 0.0))})\n")

        # Per-scenario detail
        lines.append("## 分场景对比")
        for sid in scenario_keys:
            p = pre.get(sid, {})
            q = post.get(sid, {})
            lines.append(f"### 场景：{sid}")
            lines.append(f"- 轮次数: 前 {p.get('rounds', 0)} | 后 {q.get('rounds', 0)}")
            lines.append(f"- 帖子数: 前 {p.get('posts', 0)} | 后 {q.get('posts', 0)} ({pct_delta(q.get('posts', 0), p.get('posts', 0))})")
            lines.append(f"- 总行为: 前 {p.get('total_actions', 0)} | 后 {q.get('total_actions', 0)} ({pct_delta(q.get('total_actions', 0), p.get('total_actions', 0))})")
            lines.append(f"- 点赞: 前 {p.get('likes', 0)} | 后 {q.get('likes', 0)} ({pct_delta(q.get('likes', 0), p.get('likes', 0))})")
            lines.append(f"- 评论: 前 {p.get('comments', 0)} | 后 {q.get('comments', 0)} ({pct_delta(q.get('comments', 0), p.get('comments', 0))})")
            lines.append(f"- 平均行为/帖: 前 {p.get('avg_actions_per_post', 0):.2f} | 后 {q.get('avg_actions_per_post', 0):.2f} ({pct_delta(q.get('avg_actions_per_post', 0.0), p.get('avg_actions_per_post', 0.0))})\n")

        # Top improvements by total actions
        lines.append("## 提升排名（按总行为数的相对增幅）")
        ranking: List[Tuple[str, float]] = []
        for sid in scenario_keys:
            base = pre.get(sid, {}).get("total_actions", 0)
            newv = post.get(sid, {}).get("total_actions", 0)
            ratio = (newv - base) / base if base else float("inf") if newv > 0 else 0.0
            ranking.append((sid, ratio))
        ranking.sort(key=lambda x: (x[1] if x[1] != float('inf') else 1e9), reverse=True)
        for i, (sid, r) in enumerate(ranking, 1):
            if r == float("inf"):
                r_display = "从0到正数 (∞)"
            else:
                r_display = f"{r:+.1%}"
            lines.append(f"- 第{i}名：{sid}（总行为增幅 {r_display}）")

        # Attach EvalAgent high-level if provided
        if evaluation:
            lines.append("\n## EvalAgent 综合评估摘要")
            oa = evaluation.get("overall_assessment", {})
            eff = evaluation.get("effect_metrics", {})
            cog = evaluation.get("cognitive_metrics", {})
            lines.append(f"- 综合得分: {oa.get('composite_score', 0.0):.3f} 等级: {oa.get('grade', 'N/A')} 结论: {oa.get('assessment','N/A')}")
            lines.append(f"- 效果: CTR {eff.get('ctr',0.0):.3f} | 深度参与 {eff.get('deep_engagement_index',0.0):.3f} | 传播 {eff.get('propagation_influence',0.0):.3f}")
            lines.append(f"- 认知: 一致性 {cog.get('consistency_score',0.0):.3f} | 极性 {cog.get('sentiment_polarity_strength',0.0):.3f} | 转化 {cog.get('behavior_conversion_rate',0.0):.3f}")

        report_md = "\n".join(lines) + "\n"
        out_path = self.out_dir / "multi_scenario_comparison_report.md"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report_md)
        except Exception as e:
            print(f"⚠️ 写入对比报告失败: {e}")
        return str(out_path)

    # --------------------- Graceful Shutdown ---------------------
    async def shutdown(self):
        """Gracefully close engines and agents to avoid event loop warnings on Windows."""
        # Close simulation engine (closes AsyncOpenAI client)
        try:
            if self.sim_engine is not None:
                await self.sim_engine.close()
        except Exception as e:
            print(f"⚠️ 关闭模拟引擎失败: {e}")

        # Shutdown DISTAgent
        try:
            if self.dist_agent is not None:
                await self.dist_agent.shutdown()
        except Exception as e:
            print(f"⚠️ 关闭DISTAgent失败: {e}")


async def main():
    print("🧠 多场景分发 + EvalAgent智能优化 (demo_2)")
    demo = MultiScenarioSmartDemo()
    try:
        # Load data
        demo.posts_data = demo.load_posts_from_csv(
            csv_path="Data/integrated_data/XMSU7D_integrated_articles.csv", max_posts=12
        )
        if not demo.posts_data:
            demo.posts_data = demo._load_sample_posts()
        demo.initialize_users(total_users=24)

        # Initialize systems
        await demo.initialize()

        # Run pipeline
        await demo.run(pre_eval_cycles=1, post_eval_cycles=1)
    finally:
        # Ensure graceful shutdown to avoid 'Event loop is closed' warnings
        try:
            await demo.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
