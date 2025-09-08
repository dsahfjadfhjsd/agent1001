# -*- coding: utf-8 -*-
"""
DISTAgent与原有DistributionAgent的集成桥接
提供兼容性接口和平滑迁移支持
"""

import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# 导入原有和新的模块
from .distagent_framework import DISTAgent, create_distagent, ContentDistributionTask
from .distribution_agent import DistributionAgent

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """桥接配置"""
    use_distagent: bool = True  # 是否使用新的DISTAgent
    fallback_to_legacy: bool = True  # 是否回退到原有实现
    performance_comparison: bool = True  # 是否进行性能比较
    migration_mode: str = 'gradual'  # 迁移模式：gradual, immediate, hybrid


class DistributionAgentBridge:
    """分发代理桥接类"""
    
    def __init__(self, config: BridgeConfig = None, legacy_args: Dict[str, Any] = None):
        self.config = config or BridgeConfig()
        self.legacy_args = legacy_args or {}
        
        # 初始化代理实例
        self.distagent = None
        self.legacy_agent = None
        
        # 性能统计
        self.performance_stats = {
            'distagent': {'calls': 0, 'success': 0, 'avg_time': 0.0},
            'legacy': {'calls': 0, 'success': 0, 'avg_time': 0.0}
        }
        
        logger.info(f"分发代理桥接初始化，模式: {self.config.migration_mode}")
    
    async def initialize(self):
        """初始化桥接系统"""
        try:
            if self.config.use_distagent:
                await self._initialize_distagent()
            
            if self.config.fallback_to_legacy:
                await self._initialize_legacy_agent()
                
            logger.info("桥接系统初始化完成")
            
        except Exception as e:
            logger.error(f"桥接系统初始化失败: {e}")
            raise
    
    async def _initialize_distagent(self):
        """初始化DISTAgent"""
        try:
            distagent_config = {
                'cognitive_config': {
                    'model_name': 'llama-3-8b',
                    'use_distributed': False,
                    'api_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                    'api_key': os.getenv('OPENAI_API_KEY', 'demo-key')
                },
                'memory_config': {
                    'use_distributed_db': False,
                    'vector_dim': 768,
                    'use_kafka': False
                },
                'tool_config': {
                    'sentiment_model': 'roberta-base',
                    'use_distributed': False
                },
                'action_config': {
                    'mcts_iterations': 50,
                    'use_distributed': False,
                    'use_rl': True
                },
                'evaluation_config': {
                    'learning_rate': 0.01,
                    'weight_strategy': 'adaptive'
                },
                'global_config': {
                    'agent_id': 'bridge_distagent'
                }
            }
            
            self.distagent = create_distagent(distagent_config)
            await self.distagent.initialize()
            
            logger.info("DISTAgent初始化成功")
            
        except Exception as e:
            logger.error(f"DISTAgent初始化失败: {e}")
            if not self.config.fallback_to_legacy:
                raise
    
    async def _initialize_legacy_agent(self):
        """初始化原有DistributionAgent"""
        try:
            # 使用传入的参数或默认参数初始化原有agent
            self.legacy_agent = DistributionAgent(**self.legacy_args)
            
            logger.info("原有DistributionAgent初始化成功")
            
        except Exception as e:
            logger.error(f"原有DistributionAgent初始化失败: {e}")
            raise
    
    async def generate_round_distribution(self, round_number: int, 
                                        posts_per_round: int = 5, 
                                        users_per_post: int = 10, 
                                        hot_post_ratio: float = 0.4) -> Dict[str, Any]:
        """生成分发轮次（兼容原有接口）"""
        
        method_used = None
        result = None
        start_time = datetime.now()
        
        try:
            # 根据配置选择使用的方法
            if self._should_use_distagent():
                method_used = 'distagent'
                result = await self._generate_with_distagent(
                    round_number, posts_per_round, users_per_post, hot_post_ratio
                )
            else:
                method_used = 'legacy'
                result = await self._generate_with_legacy(
                    round_number, posts_per_round, users_per_post, hot_post_ratio
                )
            
            # 记录性能统计
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(method_used, True, processing_time)
            
            # 添加桥接元数据
            result['bridge_metadata'] = {
                'method_used': method_used,
                'processing_time': processing_time,
                'bridge_version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 如果DISTAgent失败且允许回退，尝试使用原有方法
            if method_used == 'distagent' and self.config.fallback_to_legacy and self.legacy_agent:
                logger.warning(f"DISTAgent失败，回退到原有实现: {e}")
                try:
                    result = await self._generate_with_legacy(
                        round_number, posts_per_round, users_per_post, hot_post_ratio
                    )
                    self._update_performance_stats('legacy', True, processing_time)
                    
                    result['bridge_metadata'] = {
                        'method_used': 'legacy_fallback',
                        'original_error': str(e),
                        'processing_time': processing_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return result
                    
                except Exception as fallback_error:
                    logger.error(f"回退也失败: {fallback_error}")
                    self._update_performance_stats(method_used or 'unknown', False, processing_time)
                    raise
            else:
                self._update_performance_stats(method_used or 'unknown', False, processing_time)
                raise
    
    async def _generate_with_distagent(self, round_number: int, 
                                     posts_per_round: int, 
                                     users_per_post: int, 
                                     hot_post_ratio: float) -> Dict[str, Any]:
        """使用DISTAgent生成分发"""
        
        if not self.distagent:
            raise RuntimeError("DISTAgent未初始化")
        
        # 构造DISTAgent任务格式
        task = ContentDistributionTask(
            task_id=f'bridge_task_round_{round_number}',
            content_data={
                'text': f'第{round_number}轮分发任务',
                'type': 'distribution_round',
                'round_number': round_number
            },
            target_users=[f'user_{i:04d}' for i in range(users_per_post * posts_per_round)],
            distribution_params={
                'posts_per_round': posts_per_round,
                'users_per_post': users_per_post,
                'hot_post_ratio': hot_post_ratio,
                'priority': 'medium'
            },
            priority='medium',
            created_at=datetime.now()
        )
        
        # 提交任务并等待结果
        task_id = await self.distagent.submit_distribution_task(task)
        
        # 等待任务完成（简化版，实际可能需要更复杂的等待逻辑）
        await asyncio.sleep(2)
        
        # 获取任务结果
        task_result = self.distagent.get_task_status(task_id)
        
        # 转换为兼容格式
        return self._convert_distagent_result_to_legacy_format(task_result, round_number)
    
    async def _generate_with_legacy(self, round_number: int, 
                                  posts_per_round: int, 
                                  users_per_post: int, 
                                  hot_post_ratio: float) -> Dict[str, Any]:
        """使用原有DistributionAgent生成分发"""
        
        if not self.legacy_agent:
            raise RuntimeError("原有DistributionAgent未初始化")
        
        # 调用原有方法
        return self.legacy_agent.generate_round_distribution(
            round_number, posts_per_round, users_per_post, hot_post_ratio
        )
    
    def _convert_distagent_result_to_legacy_format(self, distagent_result: Dict[str, Any], 
                                                 round_number: int) -> Dict[str, Any]:
        """将DISTAgent结果转换为原有格式"""
        
        # 构造兼容的返回格式
        legacy_format = {
            'round_number': round_number,
            'status': distagent_result.get('status', 'unknown'),
            'distribution_strategy': 'distagent_optimized',
            'selected_posts': [],
            'user_assignments': {},
            'distribution_metadata': {
                'source': 'distagent',
                'confidence': 0.85,
                'optimization_applied': True
            }
        }
        
        # 从DISTAgent结果中提取信息
        if 'cognitive_analysis' in distagent_result:
            legacy_format['cognitive_insights'] = distagent_result['cognitive_analysis']
        
        if 'action_strategy' in distagent_result:
            strategy = distagent_result['action_strategy']
            legacy_format['user_assignments'] = strategy.get('user_assignments', {})
            legacy_format['platform_strategy'] = strategy.get('platform_selection', {})
        
        if 'distribution_result' in distagent_result:
            dist_result = distagent_result['distribution_result']
            legacy_format['execution_status'] = dist_result.get('execution_status')
            legacy_format['distribution_id'] = dist_result.get('distribution_id')
        
        return legacy_format
    
    def _should_use_distagent(self) -> bool:
        """判断是否应该使用DISTAgent"""
        
        if not self.distagent:
            return False
        
        if self.config.migration_mode == 'immediate':
            return True
        elif self.config.migration_mode == 'gradual':
            # 逐步迁移：根据成功率决定
            distagent_stats = self.performance_stats['distagent']
            if distagent_stats['calls'] == 0:
                return True  # 首次尝试
            
            success_rate = distagent_stats['success'] / distagent_stats['calls']
            return success_rate > 0.7  # 成功率超过70%才使用
        elif self.config.migration_mode == 'hybrid':
            # 混合模式：交替使用
            total_calls = sum(stats['calls'] for stats in self.performance_stats.values())
            return total_calls % 2 == 0
        
        return False
    
    def _update_performance_stats(self, method: str, success: bool, processing_time: float):
        """更新性能统计"""
        if method not in self.performance_stats:
            self.performance_stats[method] = {'calls': 0, 'success': 0, 'avg_time': 0.0}
        
        stats = self.performance_stats[method]
        stats['calls'] += 1
        
        if success:
            stats['success'] += 1
        
        # 更新平均时间
        stats['avg_time'] = (stats['avg_time'] * (stats['calls'] - 1) + processing_time) / stats['calls']
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """获取性能对比"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'migration_mode': self.config.migration_mode,
            'statistics': self.performance_stats.copy()
        }
        
        # 计算对比指标
        distagent_stats = self.performance_stats.get('distagent', {})
        legacy_stats = self.performance_stats.get('legacy', {})
        
        if distagent_stats.get('calls', 0) > 0 and legacy_stats.get('calls', 0) > 0:
            comparison['performance_improvement'] = {
                'success_rate_improvement': (
                    distagent_stats['success'] / distagent_stats['calls'] - 
                    legacy_stats['success'] / legacy_stats['calls']
                ),
                'time_improvement': (
                    legacy_stats['avg_time'] - distagent_stats['avg_time']
                ) / legacy_stats['avg_time'] if legacy_stats['avg_time'] > 0 else 0
            }
        
        return comparison
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """获取桥接状态"""
        return {
            'bridge_active': True,
            'distagent_available': self.distagent is not None,
            'legacy_agent_available': self.legacy_agent is not None,
            'migration_mode': self.config.migration_mode,
            'total_calls': sum(stats['calls'] for stats in self.performance_stats.values()),
            'performance_stats': self.performance_stats,
            'last_update': datetime.now().isoformat()
        }
    
    async def update_distribution_result(self, distribution_id: str, result_data: Dict[str, Any]):
        """更新分发结果（兼容原有接口）"""
        try:
            # 尝试更新两个系统的结果
            if self.distagent:
                # DISTAgent的结果更新逻辑
                logger.debug(f"更新DISTAgent分发结果: {distribution_id}")
            
            if self.legacy_agent and hasattr(self.legacy_agent, 'update_distribution_result'):
                # 原有系统的结果更新
                self.legacy_agent.update_distribution_result(distribution_id, result_data)
                
        except Exception as e:
            logger.error(f"更新分发结果失败: {e}")
    
    async def integrate_evaluation_result(self, evaluation_data: Dict[str, Any]):
        """集成评估结果（兼容原有接口）"""
        try:
            # 更新DISTAgent的评估
            if self.distagent and self.distagent.evaluation_module:
                await self.distagent.evaluation_module.feedback_processor.publish_feedback(
                    evaluation_data
                )
            
            # 更新原有系统的评估
            if self.legacy_agent and hasattr(self.legacy_agent, 'integrate_evaluation_result'):
                self.legacy_agent.integrate_evaluation_result(evaluation_data)
                
        except Exception as e:
            logger.error(f"集成评估结果失败: {e}")
    
    async def shutdown(self):
        """关闭桥接系统"""
        logger.info("开始关闭桥接系统...")
        
        if self.distagent:
            await self.distagent.shutdown()
        
        # 原有agent通常没有async shutdown方法
        # if self.legacy_agent and hasattr(self.legacy_agent, 'shutdown'):
        #     await self.legacy_agent.shutdown()
        
        logger.info("桥接系统已关闭")


# 便捷工厂函数
def create_bridge_agent(legacy_args: Dict[str, Any] = None, 
                       bridge_config: BridgeConfig = None) -> DistributionAgentBridge:
    """创建桥接代理"""
    
    default_legacy_args = {
        'batch_id': 'bridge_batch',
        'num_rounds': 10,
        'posts_per_round': 5,
        'users_per_post': 10,
        'hot_post_ratio': 0.4
    }
    
    merged_legacy_args = {**default_legacy_args, **(legacy_args or {})}
    
    return DistributionAgentBridge(
        config=bridge_config or BridgeConfig(),
        legacy_args=merged_legacy_args
    )


# 兼容性装饰器
def bridge_compatible(func):
    """使函数兼容桥接模式"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"桥接兼容性调用失败: {e}")
            # 可以在这里添加回退逻辑
            raise
    return wrapper


# 使用示例
if __name__ == "__main__":
    async def main():
        # 创建桥接配置
        bridge_config = BridgeConfig(
            use_distagent=True,
            fallback_to_legacy=True,
            migration_mode='gradual'
        )
        
        # 原有DistributionAgent的初始化参数
        legacy_args = {
            'batch_id': 'test_batch',
            'num_rounds': 5
        }
        
        # 创建桥接代理
        bridge_agent = create_bridge_agent(legacy_args, bridge_config)
        
        try:
            # 初始化
            await bridge_agent.initialize()
            
            # 使用兼容的接口生成分发
            result = await bridge_agent.generate_round_distribution(
                round_number=1,
                posts_per_round=3,
                users_per_post=5,
                hot_post_ratio=0.3
            )
            
            print(f"分发结果: {result}")
            
            # 获取性能对比
            performance = bridge_agent.get_performance_comparison()
            print(f"性能对比: {performance}")
            
            # 获取桥接状态
            status = bridge_agent.get_bridge_status()
            print(f"桥接状态: {status}")
            
        except Exception as e:
            print(f"桥接测试失败: {e}")
        
        finally:
            await bridge_agent.shutdown()
    
    # 运行测试
    asyncio.run(main())
