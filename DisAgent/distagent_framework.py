# -*- coding: utf-8 -*-
"""
DISTAgent 框架主类
整合认知基础、记忆、工具、行动和评估反馈五个核心模块
"""

import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# 导入各个模块
from .cognitive_foundation import CognitiveFoundationModel, create_cognitive_foundation_model
from .memory_module import MemoryModule, create_memory_module
from .tool_module import ToolModule, create_tool_module
from .action_module import ActionModule, create_action_module
from .evaluation_module import EvaluationModule, create_evaluation_module
from EvalAgent import EvalAgent, create_eval_agent, DistributionExpectation

logger = logging.getLogger(__name__)


@dataclass
class DISTAgentConfig:
    """DISTAgent配置"""
    # 认知基础配置
    cognitive_config: Dict[str, Any]
    # 记忆模块配置
    memory_config: Dict[str, Any]
    # 工具模块配置
    tool_config: Dict[str, Any]
    # 行动模块配置
    action_config: Dict[str, Any]
    # 评估模块配置
    evaluation_config: Dict[str, Any]
    # EvalAgent配置
    eval_agent_config: Dict[str, Any]
    # 全局配置
    global_config: Dict[str, Any]


@dataclass
class ContentDistributionTask:
    """内容分发任务"""
    task_id: str
    content_data: Dict[str, Any]
    target_users: List[str]
    distribution_params: Dict[str, Any]
    priority: str  # high, medium, low
    created_at: datetime


class DISTAgent:
    """DISTAgent框架主类"""
    
    def __init__(self, config: DISTAgentConfig):
        self.config = config
        self.agent_id = self.config.global_config.get('agent_id', 'distagent_001')
        
        # 初始化各个模块
        self.cognitive_foundation = None
        self.memory_module = None
        self.tool_module = None
        self.action_module = None
        self.evaluation_module = None
        self.eval_agent = None
        
        # 任务队列和状态
        self.task_queue = asyncio.Queue() if asyncio else []
        self.active_tasks = {}
        self.task_history = []
        
        # 性能指标
        self.performance_metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_response_time': 0.0,
            'last_optimization_time': None
        }
        
        # 系统状态
        self.is_running = False
        self.last_health_check = None
        
        logger.info(f"DISTAgent {self.agent_id} 初始化开始")
    
    async def initialize(self):
        """异步初始化所有模块"""
        try:
            logger.info("开始初始化DISTAgent模块...")
            
            # 初始化认知基础模块
            logger.info("初始化认知基础模块...")
            self.cognitive_foundation = create_cognitive_foundation_model(
                self.config.cognitive_config
            )
            await self.cognitive_foundation.initialize()
            
            # 初始化记忆模块
            logger.info("初始化记忆模块...")
            self.memory_module = create_memory_module(
                self.config.memory_config
            )
            await self.memory_module.initialize()
            
            # 初始化工具模块
            logger.info("初始化工具模块...")
            self.tool_module = create_tool_module(
                self.config.tool_config
            )
            await self.tool_module.initialize()
            
            # 初始化行动模块
            logger.info("初始化行动模块...")
            self.action_module = create_action_module(
                self.config.action_config
            )
            await self.action_module.initialize()
            
            # 初始化评估模块
            logger.info("初始化评估模块...")
            self.evaluation_module = create_evaluation_module(
                self.config.evaluation_config
            )
            
            # 初始化EvalAgent
            logger.info("初始化EvalAgent评估智能体...")
            self.eval_agent = create_eval_agent(
                self.config.eval_agent_config
            )
            
            self.is_running = True
            self.last_health_check = datetime.now()
            
            logger.info("DISTAgent所有模块初始化完成")
            
        except Exception as e:
            logger.error(f"DISTAgent初始化失败: {e}")
            raise
    
    async def submit_distribution_task(self, task: ContentDistributionTask) -> str:
        """提交内容分发任务"""
        try:
            if not self.is_running:
                raise RuntimeError("DISTAgent未正确初始化")
            
            # 将任务加入队列
            if asyncio:
                await self.task_queue.put(task)
            
            # 记录任务
            self.active_tasks[task.task_id] = {
                'task': task,
                'status': 'queued',
                'submitted_at': datetime.now(),
                'progress': 0.0
            }
            
            logger.info(f"任务 {task.task_id} 已提交到队列")
            return task.task_id
            
        except Exception as e:
            logger.error(f"提交任务失败: {e}")
            raise
    
    async def process_task(self, task: ContentDistributionTask) -> Dict[str, Any]:
        """处理单个分发任务"""
        task_start_time = datetime.now()
        
        try:
            logger.info(f"开始处理任务 {task.task_id}")
            
            # 更新任务状态
            # 兼容直接调用 process_task 的场景：若未通过 submit 注册，先初始化任务状态
            if task.task_id not in self.active_tasks:
                self.active_tasks[task.task_id] = {
                    'task': task,
                    'status': 'processing',
                    'submitted_at': datetime.now(),
                    'progress': 0.0
                }
            else:
                self.active_tasks[task.task_id]['status'] = 'processing'
                self.active_tasks[task.task_id]['progress'] = 0.1
            
            # 第一步：认知分析和内容理解
            logger.debug("步骤1: 认知分析...")
            cognitive_result = await self._cognitive_analysis(task)
            self.active_tasks[task.task_id]['progress'] = 0.3
            
            # 第二步：记忆检索和用户画像
            logger.debug("步骤2: 记忆检索...")
            memory_result = await self._memory_retrieval(task, cognitive_result)
            self.active_tasks[task.task_id]['progress'] = 0.5
            
            # 第三步：工具调用和情感分析
            logger.debug("步骤3: 工具分析...")
            tool_result = await self._tool_analysis(task, cognitive_result, memory_result)
            self.active_tasks[task.task_id]['progress'] = 0.7
            
            # 第四步：决策和分发策略生成
            logger.debug("步骤4: 决策制定...")
            action_result = await self._decision_making(
                task, cognitive_result, memory_result, tool_result
            )
            self.active_tasks[task.task_id]['progress'] = 0.9
            
            # 第五步：执行分发并评估
            logger.debug("步骤5: 执行分发...")
            distribution_result = await self._execute_distribution(action_result)
            
            # 第六步：跳过EvalAgent评估 - 评估应在模拟完成后进行
            logger.debug("步骤6: 分发任务完成，等待外部模拟和评估...")
            eval_result = {'status': 'deferred', 'message': '评估将在模拟完成后进行'}
            
            # 更新任务状态
            self.active_tasks[task.task_id]['status'] = 'completed'
            self.active_tasks[task.task_id]['progress'] = 1.0
            self.active_tasks[task.task_id]['completed_at'] = datetime.now()
            
            # 计算处理时间
            processing_time = (datetime.now() - task_start_time).total_seconds()
            
            # 生成任务结果
            task_result = {
                'task_id': task.task_id,
                'status': 'success',
                'processing_time': processing_time,
                'cognitive_analysis': cognitive_result,
                'memory_insights': memory_result,
                'tool_analysis': tool_result,
                'action_strategy': action_result,
                'distribution_result': distribution_result,
                'eval_agent_assessment': eval_result,
                'completed_at': datetime.now().isoformat()
            }
            
            # 更新性能指标
            self._update_performance_metrics(processing_time, True)
            
            # 将任务移到历史记录
            task_record = self.active_tasks[task.task_id].copy()
            task_record['task_result'] = task_result
            self.task_history.append(task_record)
            del self.active_tasks[task.task_id]
            
            logger.info(f"任务 {task.task_id} 处理完成，耗时 {processing_time:.2f}s")
            return task_result
            
        except Exception as e:
            logger.error(f"处理任务 {task.task_id} 失败: {e}")
            
            # 更新任务状态为失败
            if task.task_id in self.active_tasks:
                self.active_tasks[task.task_id]['status'] = 'failed'
                self.active_tasks[task.task_id]['error'] = str(e)
            
            # 更新性能指标
            processing_time = (datetime.now() - task_start_time).total_seconds()
            self._update_performance_metrics(processing_time, False)
            
            return {
                'task_id': task.task_id,
                'status': 'failed',
                'error': str(e),
                'processing_time': processing_time
            }
    
    async def _cognitive_analysis(self, task: ContentDistributionTask) -> Dict[str, Any]:
        """认知分析步骤"""
        try:
            # 构建认知任务
            cognitive_task = {
                'content_type': 'distribution_analysis',
                'content_data': task.content_data,
                'user_contexts': [{'user_id': uid} for uid in task.target_users],
                'task_requirements': task.distribution_params
            }
            
            # 调用认知基础模块
            cognitive_result = await self.cognitive_foundation.generate_cognitive_content(
                cognitive_task
            )
            
            return {
                'content_understanding': cognitive_result.get('content_analysis', {}),
                'user_cognitive_profiles': cognitive_result.get('user_analysis', {}),
                'distribution_insights': cognitive_result.get('distribution_strategy', {}),
                'cognitive_confidence': cognitive_result.get('confidence_score', 0.7)
            }
            
        except Exception as e:
            logger.error(f"认知分析失败: {e}")
            return {'error': str(e)}
    
    async def _eval_agent_assessment(self, task: ContentDistributionTask, 
                                   distribution_result: Dict[str, Any]) -> Dict[str, Any]:
        """EvalAgent深度评估步骤"""
        try:
            if not self.eval_agent:
                logger.warning("EvalAgent未初始化，跳过深度评估")
                return {'status': 'skipped', 'reason': 'eval_agent_not_initialized'}
            
            # 构建模拟结果数据
            simulation_results = {
                'task_id': task.task_id,
                'content_data': task.content_data,
                'target_users': task.target_users,
                'distribution_result': distribution_result,
                'rounds': [{
                    'posts': [{
                        'post_id': task.task_id,
                        'content': task.content_data.get('text', ''),
                        'likes': distribution_result.get('engagement_metrics', {}).get('likes', 0),
                        'comments': distribution_result.get('engagement_metrics', {}).get('comments', [])
                    }]
                }]
            }
            
            # 调用EvalAgent进行深度评估
            eval_result = await self.eval_agent.evaluate_distribution_performance(
                simulation_results=simulation_results
            )
            
            # 如果评估结果包含优化建议，应用到系统参数
            if 'optimization' in eval_result:
                await self._apply_eval_agent_optimizations(eval_result['optimization'])
            
            return eval_result
            
        except Exception as e:
            logger.error(f"EvalAgent评估失败: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _apply_eval_agent_optimizations(self, optimization: Dict[str, Any]):
        """应用EvalAgent的优化建议"""
        try:
            # 应用参数微调建议
            if 'parameter_tuning' in optimization:
                param_tuning = optimization['parameter_tuning']
                
                # 更新分发参数
                if hasattr(self, 'distribution_params'):
                    self.distribution_params.update(param_tuning)
                else:
                    self.distribution_params = param_tuning
                
                logger.info(f"应用参数优化: {param_tuning}")
            
            # 应用提示优化
            if 'prompt_optimization' in optimization:
                prompt_opt = optimization['prompt_optimization']
                
                # 更新认知基础模块的提示
                if self.cognitive_foundation and 'distribution_prompt' in prompt_opt:
                    await self.cognitive_foundation.update_system_prompt(
                        prompt_opt['distribution_prompt']
                    )
                
                logger.info("应用提示优化完成")
            
            # 应用策略调整
            if 'strategy_adjustments' in optimization:
                strategy = optimization['strategy_adjustments']
                
                # 更新行动模块的策略权重
                if self.action_module:
                    self.action_module.update_strategy_weights(strategy)
                
                logger.info(f"应用策略调整: {strategy}")
                
        except Exception as e:
            logger.error(f"应用优化建议失败: {e}")
    
    async def _memory_retrieval(self, task: ContentDistributionTask, 
                               cognitive_result: Dict[str, Any]) -> Dict[str, Any]:
        """记忆检索步骤"""
        try:
            memory_results = {}
            
            # 为每个目标用户检索记忆
            for user_id in task.target_users:
                # 检索用户历史记忆
                user_memories = await self.memory_module.retrieve_memories(
                    query=f"user:{user_id} content_type:{task.content_data.get('type', 'general')}",
                    user_id=user_id,
                    memory_type='both'
                )
                
                memory_results[user_id] = {
                    'historical_interactions': user_memories.get('memories', []),
                    'preference_profile': user_memories.get('user_profile', {}),
                    'interaction_patterns': user_memories.get('patterns', {}),
                    'relevance_score': user_memories.get('relevance_score', 0.5)
                }
            
            # 存储当前任务上下文到短期记忆
            await self.memory_module.store_memory(
                memory_type='short_term',
                content=f"分发任务: {task.task_id}",
                user_id='system',
                metadata={
                    'task_id': task.task_id,
                    'content_type': task.content_data.get('type'),
                    'target_users': task.target_users,
                    'cognitive_insights': cognitive_result
                }
            )
            
            return {
                'user_memory_profiles': memory_results,
                'context_stored': True,
                'memory_confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return {'error': str(e)}
    
    async def _tool_analysis(self, task: ContentDistributionTask,
                           cognitive_result: Dict[str, Any],
                           memory_result: Dict[str, Any]) -> Dict[str, Any]:
        """工具分析步骤"""
        try:
            # 情感分析
            sentiment_result = await self.tool_module.analyze_sentiment(
                text=task.content_data.get('text', ''),
                context={'task_id': task.task_id}
            )
            
            # 传播分析（如果有社交网络信息）
            propagation_result = {}
            if task.target_users:
                propagation_result = await self.tool_module.analyze_propagation(
                    user_ids=task.target_users,
                    content_metadata=task.content_data
                )
            
            return {
                'sentiment_analysis': sentiment_result,
                'propagation_analysis': propagation_result,
                'tool_confidence': 0.85,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"工具分析失败: {e}")
            return {'error': str(e)}
    
    async def _decision_making(self, task: ContentDistributionTask,
                              cognitive_result: Dict[str, Any],
                              memory_result: Dict[str, Any],
                              tool_result: Dict[str, Any]) -> Dict[str, Any]:
        """决策制定步骤"""
        try:
            # 构建决策上下文
            decision_context = {
                'task_data': task,
                'cognitive_insights': cognitive_result,
                'memory_profiles': memory_result,
                'tool_analysis': tool_result,
                'available_platforms': ['twitter', 'weibo', 'wechat']
            }
            
            # MCTS决策
            decision_result = await self.action_module.make_decision(decision_context)
            
            return {
                'optimal_strategy': decision_result.get('strategy', {}),
                'user_assignments': decision_result.get('user_assignments', {}),
                'platform_selection': decision_result.get('platform_strategy', {}),
                'timing_strategy': decision_result.get('timing', {}),
                'confidence_score': decision_result.get('confidence', 0.7),
                'alternative_strategies': decision_result.get('alternatives', []),
                'generation_reason': decision_result.get('generation_reason', {})
            }
            
        except Exception as e:
            logger.error(f"决策制定失败: {e}")
            return {'error': str(e)}
    
    async def _execute_distribution(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """执行分发步骤"""
        try:
            # 执行分发策略
            execution_result = await self.action_module.execute_action(action_result)
            
            # 开始评估反馈收集
            distribution_data = {
                'distribution_id': execution_result.get('distribution_id'),
                'content_sentiment': action_result.get('optimal_strategy', {}).get('sentiment'),
                'target_attributes': action_result.get('optimal_strategy', {})
            }
            
            # 异步启动评估
            asyncio.create_task(self._start_evaluation_tracking(distribution_data))
            
            return {
                'execution_status': execution_result.get('status', 'unknown'),
                'distribution_id': execution_result.get('distribution_id'),
                # 不再暴露平台维度；提供一次性分发摘要
                'summary': execution_result.get('summary'),
                'user_notifications': execution_result.get('notifications', []),
                'execution_timestamp': datetime.now().isoformat(),
                'evaluation_started': True
            }
            
        except Exception as e:
            logger.error(f"执行分发失败: {e}")
            return {'error': str(e)}
    
    async def _start_evaluation_tracking(self, distribution_data: Dict[str, Any]):
        """开始评估跟踪"""
        try:
            # 等待一段时间收集反馈
            await asyncio.sleep(30)  # 30秒后开始第一次评估
            
            evaluation_result = await self.evaluation_module.evaluate_distribution_performance(
                distribution_data
            )
            
            logger.info(f"分发评估结果: {evaluation_result}")
            
            # 将评估结果保存到对应的任务历史记录中
            task_id = distribution_data.get('task_id')
            if task_id:
                for task_record in self.task_history:
                    if task_record.get('task_id') == task_id:
                        task_record['evaluation_result'] = evaluation_result
                        break
            
        except Exception as e:
            logger.error(f"评估跟踪失败: {e}")
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """更新性能指标"""
        self.performance_metrics['total_tasks'] += 1
        
        if success:
            self.performance_metrics['successful_tasks'] += 1
        else:
            self.performance_metrics['failed_tasks'] += 1
        
        # 更新平均响应时间
        total_time = (self.performance_metrics['average_response_time'] * 
                     (self.performance_metrics['total_tasks'] - 1) + processing_time)
        self.performance_metrics['average_response_time'] = total_time / self.performance_metrics['total_tasks']
    
    async def run_task_processor(self):
        """运行任务处理器（后台循环）"""
        logger.info("启动DISTAgent任务处理器")
        
        while self.is_running:
            try:
                if asyncio:
                    # 等待新任务
                    task = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                    
                    # 处理任务
                    result = await self.process_task(task)
                    logger.debug(f"任务处理结果: {result.get('status')}")
                    
            except asyncio.TimeoutError:
                # 超时正常，继续循环
                continue
            except Exception as e:
                logger.error(f"任务处理器错误: {e}")
                await asyncio.sleep(1)  # 错误后等待1秒
    
    async def optimize_system_parameters(self):
        """系统参数优化"""
        try:
            if not self.task_history:
                logger.info("暂无历史任务数据，跳过参数优化")
                return
            
            # 收集最近的评估数据
            recent_evaluations = []
            for task_record in self.task_history[-10:]:  # 最近10个任务
                if 'evaluation_result' in task_record:
                    recent_evaluations.append(task_record['evaluation_result'])
            
            if not recent_evaluations:
                logger.info("暂无评估数据，跳过参数优化")
                return
            
            # 获取当前系统参数
            current_params = {
                'posts_per_round': 5,
                'users_per_post': len(self.task_history[-1]['task'].target_users) if self.task_history else 10,
                'hot_post_ratio': 0.4,
                'personalization_strength': 0.7
            }
            
            # 调用评估模块优化参数
            optimization_result = await self.evaluation_module.optimize_distribution_strategy(
                current_params, recent_evaluations
            )
            
            logger.info(f"系统参数优化完成: {optimization_result}")
            self.performance_metrics['last_optimization_time'] = datetime.now()
            
        except Exception as e:
            logger.error(f"系统参数优化失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'agent_id': self.agent_id,
            'is_running': self.is_running,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.task_history),
            'performance_metrics': self.performance_metrics,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'module_status': {
                'cognitive_foundation': self.cognitive_foundation is not None,
                'memory_module': self.memory_module is not None,
                'tool_module': self.tool_module is not None,
                'action_module': self.action_module is not None,
                'evaluation_module': self.evaluation_module is not None
            }
        }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # 在历史记录中查找
        for task_record in self.task_history:
            if task_record['task'].task_id == task_id:
                return task_record
        
        return {'error': f'未找到任务 {task_id}'}
    
    async def update_system_parameters(self, optimization_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """根据优化建议更新系统参数"""
        try:
            updated_params = {}
            
            # 更新认知基础模型参数
            if 'cognitive_params' in optimization_suggestions:
                cognitive_params = optimization_suggestions['cognitive_params']
                if hasattr(self.cognitive_foundation, 'update_system_prompt'):
                    await self.cognitive_foundation.update_system_prompt(
                        cognitive_params.get('system_prompt', '')
                    )
                updated_params['cognitive_foundation'] = cognitive_params
            
            # 更新记忆模块参数
            if 'memory_params' in optimization_suggestions:
                memory_params = optimization_suggestions['memory_params']
                # 这里可以添加记忆模块的参数更新逻辑
                updated_params['memory_module'] = memory_params
            
            # 更新工具模块参数
            if 'tool_params' in optimization_suggestions:
                tool_params = optimization_suggestions['tool_params']
                # 这里可以添加工具模块的参数更新逻辑
                updated_params['tool_module'] = tool_params
            
            logger.info(f"系统参数更新完成: {len(updated_params)} 个模块")
            return {
                'success': True,
                'updated_modules': list(updated_params.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"系统参数更新失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'modules': {}
        }
        
        try:
            # 检查各个模块
            modules = [
                ('cognitive_foundation', self.cognitive_foundation),
                ('memory_module', self.memory_module),
                ('tool_module', self.tool_module),
                ('action_module', self.action_module),
                ('evaluation_module', self.evaluation_module)
            ]
            
            for module_name, module in modules:
                if module is None:
                    health_status['modules'][module_name] = 'not_initialized'
                    health_status['overall_status'] = 'degraded'
                else:
                    # 调用模块健康检查（如果有的话）
                    if hasattr(module, 'health_check'):
                        module_health = await module.health_check()
                        health_status['modules'][module_name] = module_health
                    else:
                        health_status['modules'][module_name] = 'running'
            
            self.last_health_check = datetime.now()
            
        except Exception as e:
            health_status['overall_status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    async def shutdown(self):
        """优雅关闭"""
        logger.info("开始关闭DISTAgent...")
        
        self.is_running = False
        
        # 等待当前任务完成
        while self.active_tasks:
            logger.info(f"等待 {len(self.active_tasks)} 个任务完成...")
            await asyncio.sleep(1)
        
        # 关闭各个模块
        modules = [
            self.cognitive_foundation,
            self.memory_module,
            self.tool_module,
            self.action_module,
            self.evaluation_module
        ]
        
        for module in modules:
            if module and hasattr(module, 'shutdown'):
                try:
                    await module.shutdown()
                except Exception as e:
                    logger.error(f"模块关闭失败: {e}")
        
        logger.info("DISTAgent已关闭")


# 工厂函数
def create_distagent(config: Union[Dict[str, Any], DISTAgentConfig]) -> DISTAgent:
    """创建DISTAgent实例的工厂函数"""
    # 如果传入的是DISTAgentConfig对象，转换为字典
    if isinstance(config, DISTAgentConfig):
        config_dict = {
            'cognitive_config': config.cognitive_config,
            'memory_config': config.memory_config,
            'tool_config': config.tool_config,
            'action_config': config.action_config,
            'evaluation_config': config.evaluation_config,
        }
    else:
        config_dict = config
    
    # 默认配置
    default_config = {
        'cognitive_config': {
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 2000,
            'enable_distributed': True,
            'enable_privacy': True,
            'adaptive_weights': True,
        },
        'memory_config': {
            'vector_db_type': 'milvus',
            'enable_distributed': True,
            'enable_streaming': True,
            'enable_hierarchical': True,
        },
        'tool_config': {
            'enable_sentiment_analysis': True,
            'enable_propagation_analysis': True,
            'enable_distribution_execution': True,
        },
        'action_config': {
            'enable_mcts': True,
            'enable_rl_optimization': True,
            'enable_chain_reasoning': True,
        },
        'evaluation_config': {
            'enable_multidimensional_feedback': True,
            'enable_adaptive_optimization': True,
        },
        'global_config': {
            'batch_size': 32,
            'max_concurrent_tasks': 10,
            'enable_logging': True,
            'log_level': 'INFO',
            'enable_metrics': True,
            'performance_tracking': True,
        },
        'eval_agent_config': {
            'data_collection': {
                'enable_real_time': True,
                'privacy_protection': True,
                'privacy_epsilon': 1.0
            },
            'effect_analysis': {
                'enable_deep_engagement': True,
                'propagation_analysis': True,
                'enable_distributed': False
            },
            'cognitive_assessment': {
                'consistency_threshold': 0.8,
                'sentiment_analysis': True,
                'privacy_protection': True
            },
            'optimization_feedback': {
                'enable_auto_optimization': True,
                'learning_rate': 0.02,
                'prompt_learning': True,
                'adaptive_weighting': True
            }
        }
    }
    
    # 合并配置
    merged_config = {}
    for key in default_config:
        merged_config[key] = {**default_config[key], **config_dict.get(key, {})}
    
    # 创建配置对象
    config = DISTAgentConfig(
        cognitive_config=merged_config['cognitive_config'],
        memory_config=merged_config['memory_config'],
        tool_config=merged_config['tool_config'],
        action_config=merged_config['action_config'],
        evaluation_config=merged_config['evaluation_config'],
        eval_agent_config=merged_config['eval_agent_config'],
        global_config=merged_config['global_config']
    )
    
    return DISTAgent(config)


# 使用示例和测试
if __name__ == "__main__":
    async def main():
        # 创建DISTAgent配置
        config = {
            'cognitive_config': {
                'model_name': 'llama-3-8b',
                'use_distributed': False
            },
            'memory_config': {
                'use_distributed_db': False
            },
            'tool_config': {
                'use_distributed': False
            },
            'action_config': {
                'use_distributed': False
            },
            'evaluation_config': {},
            'global_config': {
                'agent_id': 'test_agent'
            }
        }
        
        # 创建DISTAgent
        agent = create_distagent(config)
        
        try:
            # 初始化
            await agent.initialize()
            
            # 启动任务处理器
            processor_task = asyncio.create_task(agent.run_task_processor())
            
            # 创建测试任务
            test_task = ContentDistributionTask(
                task_id='test_001',
                content_data={
                    'text': '这是一条测试内容，包含积极的情感和有趣的观点。',
                    'type': 'social_post',
                    'media': []
                },
                target_users=['user1', 'user2', 'user3'],
                distribution_params={
                    'priority': 'high',
                    'platforms': ['twitter', 'weibo']
                },
                priority='high',
                created_at=datetime.now()
            )
            
            # 提交任务
            task_id = await agent.submit_distribution_task(test_task)
            print(f"任务已提交: {task_id}")
            
            # 等待任务完成
            await asyncio.sleep(5)
            
            # 检查任务状态
            status = agent.get_task_status(task_id)
            print(f"任务状态: {status}")
            
            # 系统状态
            system_status = agent.get_system_status()
            print(f"系统状态: {system_status}")
            
            # 健康检查
            health = await agent.health_check()
            print(f"健康状态: {health}")
            
            # 参数优化
            await agent.optimize_system_parameters()
            
        except Exception as e:
            print(f"测试失败: {e}")
        
        finally:
            # 关闭系统
            await agent.shutdown()
    
    # 运行测试
    asyncio.run(main())
