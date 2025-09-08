# -*- coding: utf-8 -*-
"""
DISTAgent 行动模块
执行两步决策（链式推理 + 行动制定），生成并执行最优分发策略
"""

import os
import json
import logging
import asyncio
import numpy as np
import math
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# 检查依赖可用性
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray不可用，使用本地MCTS")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch不可用，强化学习功能受限")


@dataclass
class DistributionAction:
    """分发行动"""
    action_id: str
    content_ids: List[str]
    target_users: Dict[str, List[str]]  # {content_id: [user_ids]}
    timing: datetime
    platforms: List[str]
    expected_reward: float


@dataclass
class MCTSNode:
    """MCTS节点"""
    state: Dict[str, Any]
    action: Optional[DistributionAction]
    parent: Optional['MCTSNode']
    children: List['MCTSNode']
    visits: int
    total_reward: float
    ucb_score: float
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class ReinforcementLearningAgent:
    """强化学习代理"""
    
    def __init__(self, state_dim: int = 64, action_dim: int = 32, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        if TORCH_AVAILABLE:
            self.policy_net = self._build_policy_network()
            self.value_net = self._build_value_network()
            self.optimizer = optim.Adam(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                lr=lr
            )
            self.experience_buffer = []
        
    def _build_policy_network(self):
        """构建策略网络"""
        if not TORCH_AVAILABLE:
            return None
            
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_value_network(self):
        """构建价值网络"""
        if not TORCH_AVAILABLE:
            return None
            
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def select_action(self, state: np.ndarray) -> int:
        """选择行动"""
        if not TORCH_AVAILABLE or self.policy_net is None:
            return random.randint(0, self.action_dim - 1)
        
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy_net(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
        except NameError:
            return random.randint(0, self.action_dim - 1)
        
        return action
    
    def estimate_value(self, state: np.ndarray) -> float:
        """估计状态价值"""
        if not TORCH_AVAILABLE or self.value_net is None:
            return 0.5
        
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                value = self.value_net(state_tensor).item()
        except NameError:
            return 0.5
        
        return value


class MCTSEngine:
    """MCTS决策引擎"""
    
    def __init__(self, exploration_constant: float = 1.414):
        self.exploration_constant = exploration_constant
        self.rl_agent = ReinforcementLearningAgent()
    
    def search(self, root_state: Dict[str, Any], num_simulations: int = 100) -> DistributionAction:
        """MCTS搜索最优策略"""
        root = MCTSNode(
            state=root_state,
            action=None,
            parent=None,
            children=[],
            visits=0,
            total_reward=0.0,
            ucb_score=0.0
        )
        
        for _ in range(num_simulations):
            # 选择
            node = self._select_node(root)
            
            # 扩展
            if node.visits > 0 and not node.children:
                self._expand_node(node)
            
            # 模拟
            if node.children:
                child = random.choice(node.children)
                reward = self._simulate_rollout(child)
            else:
                reward = self._simulate_rollout(node)
            
            # 回传
            self._backpropagate(node, reward)
        
        # 选择最佳行动
        if root.children:
            best_child = max(root.children, key=lambda x: x.total_reward / max(1, x.visits))
            return best_child.action
        
        return None
    
    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """选择节点"""
        current = root
        
        while current.children:
            # 计算UCB分数
            for child in current.children:
                if child.visits == 0:
                    child.ucb_score = float('inf')
                else:
                    exploitation = child.total_reward / child.visits
                    exploration = self.exploration_constant * math.sqrt(
                        math.log(current.visits) / child.visits
                    )
                    child.ucb_score = exploitation + exploration
            
            # 选择UCB分数最高的子节点
            current = max(current.children, key=lambda x: x.ucb_score)
        
        return current
    
    def _expand_node(self, node: MCTSNode):
        """扩展节点"""
        possible_actions = self._generate_possible_actions(node.state)
        
        for action in possible_actions[:5]:  # 限制扩展数量
            new_state = self._apply_action(node.state, action)
            child = MCTSNode(
                state=new_state,
                action=action,
                parent=node,
                children=[],
                visits=0,
                total_reward=0.0,
                ucb_score=0.0
            )
            node.children.append(child)
    
    def _generate_possible_actions(self, state: Dict[str, Any]) -> List[DistributionAction]:
        """生成可能的行动"""
        actions = []
        
        available_contents = state.get('available_contents', [])
        available_users = state.get('available_users', [])
        
        # 生成不同的内容-用户组合
        for i, content_id in enumerate(available_contents[:3]):
            # 随机选择用户子集
            num_users = min(10, len(available_users))
            selected_users = random.sample(available_users, num_users)
            
            action = DistributionAction(
                action_id=f"action_{i}_{datetime.now().timestamp()}",
                content_ids=[content_id],
                target_users={content_id: selected_users},
                timing=datetime.now(),
                platforms=['social_media'],
                expected_reward=0.0
            )
            actions.append(action)
        
        return actions
    
    def _apply_action(self, state: Dict[str, Any], action: DistributionAction) -> Dict[str, Any]:
        """应用行动"""
        new_state = state.copy()
        
        # 更新状态
        new_state['last_action'] = action
        new_state['round_number'] = state.get('round_number', 1) + 1
        
        # 移除已使用的内容
        used_contents = set(action.content_ids)
        new_state['available_contents'] = [
            cid for cid in state.get('available_contents', [])
            if cid not in used_contents
        ]
        
        return new_state
    
    def _simulate_rollout(self, node: MCTSNode) -> float:
        """模拟展开"""
        # 使用强化学习代理估计价值
        state_vector = self._state_to_vector(node.state)
        estimated_value = self.rl_agent.estimate_value(state_vector)
        
        # 添加随机性
        noise = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, estimated_value + noise))
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """回传奖励"""
        current = node
        
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """状态转向量"""
        # 简化的状态向量化
        vector = np.zeros(64)
        
        # 内容数量特征
        vector[0] = len(state.get('available_contents', [])) / 100.0
        
        # 用户数量特征
        vector[1] = len(state.get('available_users', [])) / 1000.0
        
        # 轮次特征
        vector[2] = min(1.0, state.get('round_number', 1) / 10.0)
        
        # 历史奖励特征
        vector[3] = state.get('cumulative_reward', 0.0)
        
        # 填充其他特征
        for i in range(4, 64):
            vector[i] = random.uniform(0, 0.1)
        
        return vector


class ChainOfThought:
    """链式推理模块"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def reason_about_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """对策略进行推理"""
        try:
            # 构建推理提示
            reasoning_prompt = self._build_reasoning_prompt(context)
            
            if self.llm_client:
                # 使用LLM进行推理
                reasoning_result = self.llm_client.invoke(reasoning_prompt)
                return self._parse_reasoning_result(reasoning_result)
            else:
                # 简单的基于规则的推理
                return self._rule_based_reasoning(context)
                
        except Exception as e:
            logger.error(f"链式推理失败: {e}")
            return self._default_reasoning()
    
    def _build_reasoning_prompt(self, context: Dict[str, Any]) -> str:
        """构建推理提示"""
        return f"""
        基于以下上下文信息，进行分发策略推理：
        
        当前状态:
        - 可用内容数: {len(context.get('available_contents', []))}
        - 可用用户数: {len(context.get('available_users', []))}
        - 当前轮次: {context.get('round_number', 1)}
        - 历史表现: {context.get('performance_history', {})}
        
        约束条件:
        - CTR目标: {context.get('target_ctr', 0.1)}
        - 认知一致性要求: {context.get('cognitive_consistency', 0.7)}
        
        请推理最佳的分发策略，考虑：
        1. 内容选择策略
        2. 用户匹配策略  
        3. 时机选择策略
        4. 预期效果评估
        
        返回JSON格式的推理结果。
        """
    
    def _parse_reasoning_result(self, result: str) -> Dict[str, Any]:
        """解析推理结果"""
        try:
            # 尝试解析JSON
            if isinstance(result, str):
                parsed = json.loads(result)
            else:
                parsed = result
            
            return {
                'content_strategy': parsed.get('content_strategy', 'balanced'),
                'user_strategy': parsed.get('user_strategy', 'diverse'),
                'timing_strategy': parsed.get('timing_strategy', 'immediate'),
                'expected_performance': parsed.get('expected_performance', {})
            }
            
        except Exception:
            return self._default_reasoning()
    
    def _rule_based_reasoning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于规则的推理"""
        # 简单的基于规则的策略推理
        content_count = len(context.get('available_contents', []))
        user_count = len(context.get('available_users', []))
        round_number = context.get('round_number', 1)
        
        if round_number <= 2:
            strategy = 'exploration'
        elif content_count > user_count:
            strategy = 'content_focused'
        else:
            strategy = 'user_focused'
        
        return {
            'content_strategy': strategy,
            'user_strategy': 'balanced',
            'timing_strategy': 'immediate',
            'expected_performance': {
                'ctr': 0.1,
                'engagement': 0.7
            }
        }
    
    def _default_reasoning(self) -> Dict[str, Any]:
        """默认推理结果"""
        return {
            'content_strategy': 'balanced',
            'user_strategy': 'diverse',
            'timing_strategy': 'immediate',
            'expected_performance': {
                'ctr': 0.05,
                'engagement': 0.5
            }
        }


class ActionExecutor:
    """行动执行器"""
    
    def __init__(self, tool_module=None):
        self.tool_module = tool_module
        self.execution_history = []
    
    async def execute_action(self, action: DistributionAction) -> Dict[str, Any]:
        """执行分发行动"""
        try:
            execution_start = datetime.now()
            
            # 记录执行历史
            execution_record = {
                'action_id': action.action_id,
                'start_time': execution_start,
                'status': 'executing'
            }
            self.execution_history.append(execution_record)
            
            # 执行内容分发
            if self.tool_module:
                results = {}
                reach_sum = 0
                for content_id in action.content_ids:
                    target_users = action.target_users.get(content_id, [])
                    # 使用工具模块执行分发（内部仍可多平台），但这里只做聚合，不暴露平台维度
                    dist_res = await self.tool_module.execute_distribution(
                        content=f"content_{content_id}",
                        target_users=target_users,
                        platforms=action.platforms
                    )
                    results[content_id] = dist_res
                    try:
                        for pr in (dist_res.get('distribution_results', {}) or {}).values():
                            reach_sum += int(pr.get('estimated_reach', 0))
                    except Exception:
                        pass
                # 聚合为一次性分发摘要
                total_targets = sum(len(v) for v in (action.target_users or {}).values())
                summary = {
                    'content_ids': action.content_ids,
                    'total_target_users': total_targets,
                    'estimated_reach': reach_sum if reach_sum > 0 else None,
                    'timestamp': datetime.now().isoformat()
                }
                # 更新执行记录
                execution_record['status'] = 'completed'
                execution_record['end_time'] = datetime.now()
                execution_record['summary'] = summary
                
                return {
                    'success': True,
                    'status': 'success',
                    'action_id': action.action_id,
                    'distribution_id': action.action_id,
                    'summary': summary,
                    'notifications': [],
                    'execution_time': (execution_record['end_time'] - execution_start).total_seconds()
                }
            else:
                # 模拟执行
                await asyncio.sleep(0.1)  # 模拟执行时间
                total_targets = sum(len(v) for v in (action.target_users or {}).values())
                summary = {
                    'content_ids': action.content_ids,
                    'total_target_users': total_targets,
                    'estimated_reach': None,
                    'timestamp': datetime.now().isoformat()
                }
                return {
                    'success': True,
                    'status': 'success',
                    'action_id': action.action_id,
                    'distribution_id': action.action_id,
                    'summary': summary,
                    'notifications': [],
                    'execution_time': 0.1
                }
                
        except Exception as e:
            logger.error(f"执行行动失败: {e}")
            execution_record['status'] = 'failed'
            execution_record['error'] = str(e)
            
            return {
                'success': False,
                'action_id': action.action_id,
                'error': str(e)
            }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history.copy()


class ActionModule:
    """行动模块主类"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None, tool_module=None):
        self.config = config or {}
        
        # 初始化组件
        self.mcts_engine = MCTSEngine(
            exploration_constant=self.config.get('exploration_constant', 1.414)
        )
        
        self.chain_of_thought = ChainOfThought(llm_client)
        self.action_executor = ActionExecutor(tool_module)
        self.strategy_cache = {}
        # Platform control
        self.single_platform_mode = bool(self.config.get('single_platform_mode', False))
        self.default_platform = self.config.get('default_platform', 'social')
        
        # 分布式MCTS（如果Ray可用）
        if RAY_AVAILABLE:
            self.distributed_mcts = None  # 可以在需要时初始化
        
        logger.info("行动模块初始化完成")
    
    async def initialize(self):
        """异步初始化方法"""
        try:
            logger.info("开始初始化行动模块...")
            
            # 初始化MCTS引擎
            self.mcts_engine = MCTSEngine()
            
            # 初始化强化学习代理
            if TORCH_AVAILABLE:
                logger.info("初始化强化学习组件...")
            else:
                logger.info("PyTorch不可用，使用规则决策")
                
            logger.info("行动模块初始化完成")
            
        except Exception as e:
            logger.error(f"行动模块初始化失败: {e}")
            logger.warning("使用降级模式：简单规则决策")

    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """根据上下文生成分发策略，返回包含 strategy/user_assignments 等键的字典"""
        try:
            # 构建MCTS根状态
            task = context.get('task_data')
            available_contents = []
            available_users = []
            if task:
                # ContentDistributionTask dataclass
                posts = task.content_data.get('posts', []) if hasattr(task, 'content_data') else []
                available_contents = [str(p.get('post_id') or p.get('id') or f"content_{i}") for i, p in enumerate(posts)]
                available_users = list(task.target_users) if hasattr(task, 'target_users') else []
            else:
                available_contents = context.get('available_contents', [])
                available_users = context.get('available_users', [])

            # 生成原因（默认OK）
            generation_reason = {
                'reason_code': 'ok',
                'reason_message': 'strategy generated successfully',
                'available_contents_count': len(available_contents),
                'available_users_count': len(available_users)
            }
            if not available_contents:
                generation_reason.update({
                    'reason_code': 'no_available_contents',
                    'reason_message': 'no available contents in task context'
                })
            elif not available_users:
                generation_reason.update({
                    'reason_code': 'no_available_users',
                    'reason_message': 'no available users in task context'
                })

            root_state = {
                'available_contents': available_contents,
                'available_users': available_users,
                'round_number': context.get('task_data').content_data.get('round', 1) if task else 1,
                'performance_history': context.get('performance_history', {})
            }

            # MCTS 搜索
            simulations = self.config.get('mcts_iterations', 50)
            action = self.mcts_engine.search(root_state, num_simulations=simulations)

            # 如果没有可用行动，生成默认行动
            if action is None:
                # 标记原因：MCTS未返回行动
                generation_reason.update({
                    'reason_code': 'mcts_no_action',
                    'reason_message': 'MCTS returned no action; fallback to default'
                })
                action = self._generate_default_action(root_state)

            # 用户分配：从状态里取已分配的 target_users
            user_assignments = action.target_users if action and action.target_users else {
                action.content_ids[0]: available_users[: min(10, len(available_users))]
            }

            # 平台维度彻底移除：不再返回 platform_strategy

            # 置信度估计（简化）
            confidence = 0.75

            strategy = {
                'selected_contents': action.content_ids,
                'sentiment': context.get('cognitive_insights', {}).get('content_understanding', {}).get('sentiment', 'neutral')
            }

            return {
                'strategy': strategy,
                'user_assignments': user_assignments,
                'platform_strategy': {},
                'timing': {'mode': 'immediate'},
                'confidence': confidence,
                'alternatives': [],
                'generation_reason': generation_reason
            }
        except Exception as e:
            logger.error(f"生成决策失败: {e}")
            # 降级返回一个最小策略
            return {
                'strategy': {'selected_contents': []},
                'user_assignments': {},
                'platform_strategy': {},
                'timing': {'mode': 'immediate'},
                'confidence': 0.5,
                'alternatives': [],
                'generation_reason': {
                    'reason_code': 'decision_exception',
                    'reason_message': str(e)
                }
            }

    async def execute_action(self, decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """根据 make_decision 的结果构造并执行实际行动，返回执行摘要"""
        try:
            # 兼容两种键名
            strategy_block = decision_result.get('strategy') or decision_result.get('optimal_strategy') or {}
            selected_contents = strategy_block.get('selected_contents', [])
            user_assignments = decision_result.get('user_assignments', {})
            platforms_map = decision_result.get('platform_strategy') or decision_result.get('platform_selection') or {}
            platforms = list(set([p for plist in platforms_map.values() for p in (plist or [])])) or ['social']

            # 构造 DistributionAction
            if not selected_contents:
                # 没有选中内容时构造一个默认内容
                selected_contents = ['default_content']
                user_assignments = {'default_content': []}

            action = DistributionAction(
                action_id=f"act_{datetime.now().timestamp()}",
                content_ids=[str(c) for c in selected_contents],
                target_users={str(k): list(v) for k, v in user_assignments.items()},
                timing=datetime.now(),
                platforms=platforms,
                expected_reward=0.0
            )

            exec_result = await self.action_executor.execute_action(action)

            # 规范化输出，兼容上层读取字段
            status = 'success' if exec_result.get('success') else 'failed'
            return {
                'status': status,
                'distribution_id': exec_result.get('action_id'),
                'platform_results': exec_result.get('results', {}),
                'notifications': [],
            }
        except Exception as e:
            logger.error(f"执行行动失败: {e}")
            return {
                'status': 'failed',
                'distribution_id': None,
                'platform_results': {},
                'notifications': [],
                'error': str(e)
            }
    
    async def execute_strategy(self, action: DistributionAction) -> Dict[str, Any]:
        """执行策略"""
        try:
            # 执行行动
            execution_result = await self.action_executor.execute_action(action)
            
            # 缓存策略
            self.strategy_cache[action.action_id] = {
                'action': action,
                'result': execution_result,
                'timestamp': datetime.now()
            }
            
            return execution_result
            
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            return {'success': False, 'error': str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        try:
            return {
                'module_name': 'ActionModule',
                'mcts_planner_ready': hasattr(self, 'mcts_planner') and self.mcts_planner is not None,
                'rl_optimizer_ready': hasattr(self, 'rl_optimizer') and self.rl_optimizer is not None,
                'action_executor_ready': hasattr(self, 'action_executor') and self.action_executor is not None,
                'chain_reasoner_ready': hasattr(self, 'chain_reasoner') and self.chain_reasoner is not None,
                'strategy_cache_size': len(self.strategy_cache) if hasattr(self, 'strategy_cache') else 0,
                'distributed_mode': self.config.get('enable_distributed', False),
                'reinforcement_learning': self.config.get('enable_reinforcement_learning', False),
                'status': 'healthy'
            }
        except Exception as e:
            return {
                'module_name': 'ActionModule',
                'status': 'error',
                'error': str(e)
            }
    
    def update_strategy_weights(self, feedback: Dict[str, Any]):
        """根据优化反馈更新策略权重"""
        try:
            # 提取策略权重调整
            strategy_adjustments = feedback.get('strategy_adjustments', {})
            
            # 更新MCTS探索常数
            if 'exploration_constant' in strategy_adjustments:
                new_exploration = strategy_adjustments['exploration_constant']
                self.mcts_engine.exploration_constant = max(0.1, min(2.0, new_exploration))
                logger.info(f"更新MCTS探索常数: {self.mcts_engine.exploration_constant}")
            
            # 更新其他策略参数
            if hasattr(self, 'strategy_weights'):
                self.strategy_weights.update(strategy_adjustments.get('weights', {}))
            else:
                self.strategy_weights = strategy_adjustments.get('weights', {})
            
            # 清空策略缓存以应用新权重
            self.strategy_cache.clear()
            
            logger.info(f"策略权重已更新: {len(strategy_adjustments)} 项调整")
            
        except Exception as e:
            logger.error(f"更新策略权重失败: {e}")
            # 不抛出异常，允许系统继续运行
    
    def _generate_default_action(self, context: Dict[str, Any]) -> DistributionAction:
        """生成默认行动"""
        available_contents = context.get('available_contents', [])
        available_users = context.get('available_users', [])
        
        # 选择一个内容和一些用户
        content_id = available_contents[0] if available_contents else 'default_content'
        selected_users = available_users[:5] if available_users else ['default_user']
        
        return DistributionAction(
            action_id=f"default_{datetime.now().timestamp()}",
            content_ids=[content_id],
            target_users={content_id: selected_users},
            timing=datetime.now(),
            platforms=['social_media'],
            expected_reward=0.1
        )
    
    def _calculate_expected_reward(self, action: DistributionAction, 
                                 reasoning: Dict[str, Any]) -> float:
        """计算期望奖励"""
        base_reward = 0.1
        
        # 基于推理结果调整奖励
        expected_perf = reasoning.get('expected_performance', {})
        ctr_bonus = expected_perf.get('ctr', 0.05) * 2.0
        engagement_bonus = expected_perf.get('engagement', 0.5) * 0.5
        
        return min(1.0, base_reward + ctr_bonus + engagement_bonus)
    
    def update_strategy_weights(self, optimization_feedback: Dict[str, Any]):
        """根据优化反馈更新策略权重"""
        try:
            # 提取策略权重调整
            strategy_adjustments = optimization_feedback.get('strategy_adjustments', {})
            
            # 更新MCTS探索常数
            if 'exploration_constant' in strategy_adjustments:
                new_exploration = strategy_adjustments['exploration_constant']
                self.mcts_engine.exploration_constant = max(0.1, min(2.0, new_exploration))
                logger.info(f"更新MCTS探索常数: {self.mcts_engine.exploration_constant}")
            
            # 更新其他策略参数
            if hasattr(self, 'strategy_weights'):
                self.strategy_weights.update(strategy_adjustments.get('weights', {}))
            else:
                self.strategy_weights = strategy_adjustments.get('weights', {})
            
            # 清空策略缓存以应用新权重
            self.strategy_cache.clear()
            
            logger.info(f"策略权重已更新: {len(strategy_adjustments)} 项调整")
            
        except Exception as e:
            logger.error(f"更新策略权重失败: {e}")
            # 不抛出异常，允许系统继续运行
    
    def get_module_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        return {
            'mcts_engine': {
                'exploration_constant': self.mcts_engine.exploration_constant,
                'rl_available': TORCH_AVAILABLE
            },
            'execution_history_length': len(self.action_executor.execution_history),
            'strategy_cache_size': len(self.strategy_cache),
            'ray_available': RAY_AVAILABLE,
            'strategy_weights': getattr(self, 'strategy_weights', {})
        }


# 工厂函数
def create_action_module(config: Dict[str, Any] = None, 
                        llm_client=None, tool_module=None) -> ActionModule:
    """创建行动模块实例"""
    return ActionModule(config, llm_client, tool_module)


# 使用示例
if __name__ == "__main__":
    # 创建行动模块
    action_module = create_action_module({
        'exploration_constant': 1.414,
        'mcts_simulations': 50
    })
    
    async def main():
        # 示例上下文
        context = {
            'available_contents': ['content_1', 'content_2', 'content_3'],
            'available_users': ['user_1', 'user_2', 'user_3', 'user_4'],
            'round_number': 2,
            'target_ctr': 0.1,
            'cognitive_consistency': 0.7
        }
        
        # 制定决策
        action, reasoning = await action_module.make_decision(context)
        print(f"决策行动: {action.action_id}")
        print(f"推理结果: {reasoning}")
        
        # 执行策略
        result = await action_module.execute_strategy(action)
        print(f"执行结果: {result}")
        
        # 获取状态
        status = action_module.get_module_status()
        print(f"模块状态: {status}")
    
    # 运行示例
    asyncio.run(main())
