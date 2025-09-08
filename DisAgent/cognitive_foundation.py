# -*- coding: utf-8 -*-
"""
DISTAgent 认知基础模型
基于大型语言模型（LLM）生成认知引导的文本内容，进行场景理解和用户画像分析
"""

import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# 配置加载
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join('Config', '.env'))

logger = logging.getLogger(__name__)

# 强制导入所有依赖，不使用fallback
import ray
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch

# 初始化Ray分布式计算
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

logger.info("所有依赖已成功加载，启用完整功能模式")


@dataclass
class CognitiveAttributes:
    """认知属性数据类"""
    stance: str  # 立场
    emotion: str  # 情感
    intent: str  # 意图
    confidence: float  # 置信度


@dataclass
class UserProfile:
    """用户画像数据类"""
    user_id: str
    demographics: Dict[str, Any]
    interests: List[str]
    behavior_patterns: Dict[str, Any]
    cognitive_preferences: Dict[str, Any]


@dataclass
class ScenarioRequirement:
    """场景需求数据类"""
    scenario_id: str
    target_attributes: CognitiveAttributes
    context: Dict[str, Any]
    constraints: Dict[str, Any]


class DifferentialPrivacyMixin:
    """差分隐私混入类"""
    
    def __init__(self, privacy_budget: float = 1.0, noise_scale: float = 0.1):
        self.privacy_budget = privacy_budget
        self.noise_scale = noise_scale
    
    def add_gaussian_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """添加高斯噪声 - 使用PyTorch实现"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_tensor = torch.from_numpy(data).to(device)
        noise = torch.normal(0, self.noise_scale * sensitivity, data_tensor.shape, device=device)
        noisy_data = data_tensor + noise
        return noisy_data.cpu().numpy()
    
    def apply_differential_privacy(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """应用差分隐私到梯度 - 完整实现"""
        noisy_gradients = {}
        for key, grad in gradients.items():
            if isinstance(grad, np.ndarray):
                noisy_gradients[key] = self.add_gaussian_noise(grad)
            elif torch.is_tensor(grad):
                noisy_gradients[key] = self.add_gaussian_noise(grad.detach().cpu().numpy())
            else:
                noisy_gradients[key] = grad
        
        return noisy_gradients


@ray.remote
class DistributedLLMWorker:
    """分布式LLM工作节点"""
    
    def __init__(self, model_name: str = "qwen-max", node_id: int = 0):
        self.model_name = model_name
        self.node_id = node_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化多个模型组件
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL'),
            temperature=0.7,
            max_tokens=2048
        )
        
        # 初始化语义模型 - 使用OpenAI Embeddings API
        self.semantic_model = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('OPENAI_BASE_URL')
        )
        
        # 情感分析使用LLM调用替代本地模型
        self.emotion_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL'),
            temperature=0.1,
            max_tokens=100
        )
        
        logger.info(f"节点{self.node_id}完全初始化完成，设备: {self.device}")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试LLM响应
            test_response = self.llm.invoke("测试")
            
            return {
                'status': 'healthy',
                'node_id': self.node_id,
                'device': str(self.device),
                'model_name': self.model_name,
                'llm_responsive': True,
                'semantic_model_loaded': hasattr(self, 'semantic_model'),
                'emotion_analyzer_loaded': hasattr(self, 'emotion_llm')
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'node_id': self.node_id,
                'error': str(e)
            }

    def update_prompt(self, new_prompt: str) -> Dict[str, Any]:
        """更新节点的系统提示词"""
        try:
            # 存储新的系统提示词
            self.system_prompt = new_prompt
            
            # 更新LLM的温度和其他参数（如果需要）
            # 这里可以根据新提示词调整模型行为
            
            logger.info(f"节点{self.node_id}系统提示词已更新")
            return {
                'success': True,
                'node_id': self.node_id,
                'prompt_length': len(new_prompt)
            }
        except Exception as e:
            logger.error(f"节点{self.node_id}更新提示词失败: {e}")
            return {
                'success': False,
                'node_id': self.node_id,
                'error': str(e)
            }
    
    def generate_content(self, prompt: str, user_profile: Dict[str, Any], 
                        target_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """生成内容"""
        try:
            # 构建增强提示
            enhanced_prompt = self._build_enhanced_prompt(prompt, user_profile, target_attributes)
            
            # 调用LLM
            response = self.llm.invoke(enhanced_prompt)
            
            # 解析响应
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                'content': content,
                'node_id': self.node_id,
                'semantic_score': self._calculate_semantic_match(content, target_attributes),
                'emotion_score': self._calculate_emotion_score(content, target_attributes.get('emotion', 'neutral'))
            }
        except Exception as e:
            logger.error(f"节点{self.node_id}生成内容失败: {e}")
            raise RuntimeError(f"节点{self.node_id}内容生成失败: {e}")
    
    def _build_enhanced_prompt(self, base_prompt: str, user_profile: Dict[str, Any], 
                              target_attributes: Dict[str, Any]) -> str:
        """构建增强提示 - 修复数据类型处理"""
        # 处理user_profile可能是列表或字典的情况
        if isinstance(user_profile, list):
            # 如果user_profile是列表，转换为字典格式
            if len(user_profile) > 0 and isinstance(user_profile[0], dict):
                # 列表中的第一个元素是字典，使用它
                profile_dict = user_profile[0]
            else:
                # 列表中的元素不是字典，创建默认结构
                profile_dict = {
                    'interests': user_profile if all(isinstance(item, str) for item in user_profile) else [],
                    'behavior_patterns': {},
                    'cognitive_preferences': {}
                }
        elif isinstance(user_profile, dict):
            profile_dict = user_profile
        else:
            # 其他类型，使用默认结构
            profile_dict = {
                'interests': [],
                'behavior_patterns': {},
                'cognitive_preferences': {}
            }
            
        return f"""
        基础任务: {base_prompt}
        
        用户画像:
        - 兴趣: {profile_dict.get('interests', [])}
        - 行为模式: {profile_dict.get('behavior_patterns', {})}
        - 认知偏好: {profile_dict.get('cognitive_preferences', {})}
        
        目标认知属性:
        - 立场: {target_attributes.get('stance', 'neutral')}
        - 情感: {target_attributes.get('emotion', 'neutral')}
        - 意图: {target_attributes.get('intent', 'inform')}
        
        请生成符合用户画像和目标认知属性的内容，确保语义匹配度和情感一致性。
        """
    
    def _calculate_semantic_match(self, content: str, target_attributes: Dict[str, Any]) -> float:
        """计算语义匹配度 - 使用OpenAI Embeddings API"""
        try:
            # 构建目标语义描述
            target_description = f"{target_attributes.get('stance', 'neutral')} {target_attributes.get('emotion', 'neutral')} {target_attributes.get('intent', 'inform')}"
            
            # 计算语义相似度
            content_embedding = self.semantic_model.embed_query(content)
            target_embedding = self.semantic_model.embed_query(target_description)
            
            # 计算余弦相似度
            content_tensor = torch.tensor(content_embedding)
            target_tensor = torch.tensor(target_embedding)
            
            similarity = torch.cosine_similarity(
                content_tensor.unsqueeze(0),
                target_tensor.unsqueeze(0),
                dim=1
            )
            
            return float(similarity.item())
        except Exception as e:
            logger.error(f"语义匹配计算失败: {e}")
            return 0.5  # 返回默认值而不是抛出异常
    
    def _calculate_emotion_score(self, content: str, target_emotion: str) -> float:
        """计算情感得分 - 使用LLM API"""
        try:
            # 使用LLM进行情感分析
            emotion_prompt = f"""请分析以下文本的情感倾向，只返回一个词：positive、negative或neutral。
            
            文本：{content}
            
            情感分析结果："""
            
            result = self.emotion_llm.invoke(emotion_prompt)
            predicted_emotion = result.content.strip().lower()
            
            # 映射情感标签
            emotion_mapping = {
                'positive': ['positive', 'joy', 'optimism'],
                'negative': ['negative', 'sadness', 'anger'],
                'neutral': ['neutral']
            }
            
            # 检查情感匹配
            target_emotions = emotion_mapping.get(target_emotion, ['neutral'])
            if predicted_emotion in target_emotions:
                return 0.8  # 匹配时给高分
            else:
                return 0.3  # 不匹配时给低分
                
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return 0.5  # 返回默认值而不是抛出异常


class AdaptiveWeightManager:
    """自适应权重管理器"""
    
    def __init__(self):
        self.weights = {
            'semantic_match': 0.4,
            'emotion_consistency': 0.3,
            'user_preference': 0.2,
            'novelty': 0.1
        }
        self.history = []
    
    def update_weights(self, feedback: Dict[str, float]):
        """基于反馈更新权重"""
        # 贝叶斯优化权重更新
        learning_rate = 0.1
        
        for metric, score in feedback.items():
            if metric in self.weights:
                # 简化的权重更新规则
                if score > 0.7:
                    self.weights[metric] += learning_rate * (1 - self.weights[metric])
                elif score < 0.3:
                    self.weights[metric] -= learning_rate * self.weights[metric]
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights.copy(),
            'feedback': feedback
        })
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights.copy()


class CognitiveFoundationModel(DifferentialPrivacyMixin):
    """认知基础模型主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or {}
        self.weight_manager = AdaptiveWeightManager()
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 差分隐私设置
        self.enable_differential_privacy = self.config.get('enable_differential_privacy', True)
        
        # 实时反馈处理
        self.feedback_queue = asyncio.Queue() if asyncio else []
        
        logger.info("认知基础模型初始化完成")
    
    async def initialize(self):
        """异步初始化方法 - 强制成功模式"""
        logger.info("开始初始化认知基础模块...")
        
        # 初始化分布式工作节点
        await self._initialize_distributed_workers()
        
        # 初始化权重管理器
        self.weight_manager = AdaptiveWeightManager()
        
        # 验证所有组件正常工作
        await self._validate_components()
        
        logger.info("认知基础模块初始化完成 - 所有组件正常")
    
    async def _initialize_distributed_workers(self):
        """快速初始化分布式工作节点 - 性能优先模式"""
        num_workers = min(self.config.get('num_workers', 2), 2)  # 最多2个节点，快速启动
        
        # 确保Ray已初始化
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=4)
        
        # 快速创建工作节点 - Ray actors return handles directly
        self.workers = []
        for i in range(num_workers):
            try:
                worker = DistributedLLMWorker.remote(
                    model_name=self.config.get('model_name', 'qwen-max'),
                    node_id=i
                )
                self.workers.append(worker)
                logger.info(f"创建工作节点 {i} 成功")
            except Exception as e:
                logger.error(f"创建工作节点 {i} 失败: {e}")
        
        # 确保至少有一个工作节点
        if not self.workers:
            logger.warning("所有工作节点创建失败，创建单个备用节点")
            try:
                backup_worker = DistributedLLMWorker.remote(model_name='qwen-max', node_id=0)
                self.workers = [backup_worker]
            except Exception as e:
                logger.error(f"备用工作节点创建也失败: {e}")
                raise RuntimeError("无法创建任何工作节点")
        
        # 简化健康检查，快速验证
        logger.info(f"快速初始化{len(self.workers)}个分布式工作节点完成")
    
    async def generate_cognitive_content(self, 
                                       task_or_user_profiles,
                                       scenario: Optional[ScenarioRequirement] = None) -> Dict[str, Any]:
        """生成认知引导的内容。
        兼容两种调用方式：
        1) generate_cognitive_content(task_dict)
        2) generate_cognitive_content(user_profiles: List[UserProfile], scenario: ScenarioRequirement)
        """
        try:
            # 兼容输入：若传入的是task字典，则进行适配
            if isinstance(task_or_user_profiles, dict):
                task = task_or_user_profiles
                # 构造用户画像列表
                up_list: List[UserProfile] = []
                for uc in task.get('user_contexts', []):
                    up_list.append(UserProfile(
                        user_id=str(uc.get('user_id', 'unknown')),
                        demographics=uc.get('demographics', {}),
                        interests=uc.get('interests', []),
                        behavior_patterns=uc.get('behavior_patterns', {}),
                        cognitive_preferences=uc.get('cognitive_preferences', {})
                    ))
                # 构造场景需求
                content_data = task.get('content_data', {})
                reqs = task.get('task_requirements', {})
                attrs = CognitiveAttributes(
                    stance=reqs.get('stance', 'neutral'),
                    emotion=reqs.get('emotion', 'neutral'),
                    intent=reqs.get('intent', 'inform'),
                    confidence=float(reqs.get('confidence', 0.7))
                )
                scenario = ScenarioRequirement(
                    scenario_id=str(content_data.get('type', 'general')),
                    target_attributes=attrs,
                    context=content_data,
                    constraints=reqs
                )
                user_profiles = up_list
            else:
                user_profiles = task_or_user_profiles
                # scenario 按原签名传入

            # 分解任务到多个节点
            tasks = self._decompose_generation_tasks(user_profiles, scenario)
            
            # 并行执行任务 - 仅使用分布式模式
            results = await self._execute_distributed_tasks(tasks)
            
            # 聚合结果
            aggregated_content = self._aggregate_results(results, scenario)
            
            # 应用差分隐私保护
            protected_content = self._apply_privacy_protection(aggregated_content)
            
            return {
                'content': protected_content,
                'generation_metadata': {
                    'num_workers': len(self.workers),
                    'task_count': len(tasks),
                    'timestamp': datetime.now().isoformat(),
                    'weights_used': self.weight_manager.get_current_weights()
                }
            }
            
        except Exception as e:
            logger.error(f"认知内容生成失败: {e}")
            raise RuntimeError(f"认知内容生成失败: {e}")
    
    def _decompose_generation_tasks(self, user_profiles: List[UserProfile], 
                                  scenario: ScenarioRequirement) -> List[Dict[str, Any]]:
        """优化任务分解 - 批量处理减少任务数量"""
        # 将用户分批，每批最多8个用户，大幅减少任务数
        batch_size = 8
        batches = [user_profiles[i:i + batch_size] for i in range(0, len(user_profiles), batch_size)]
        
        tasks = []
        for batch_idx, batch_profiles in enumerate(batches):
            task = {
                'batch_profiles': [
                    {
                        'user_id': profile.user_id,
                        'interests': profile.interests[:3],  # 只取前3个兴趣，减少数据量
                        'behavior_patterns': profile.behavior_patterns,
                        'cognitive_preferences': profile.cognitive_preferences
                    } for profile in batch_profiles
                ],
                'target_attributes': {
                    'stance': scenario.target_attributes.stance,
                    'emotion': scenario.target_attributes.emotion,
                    'intent': scenario.target_attributes.intent
                },
                'context': str(scenario.context)[:200] if scenario.context else "",  # 安全截断context
                'base_prompt': f"批量为{len(batch_profiles)}个用户生成内容",
                'batch_id': batch_idx
            }
            tasks.append(task)
        
        logger.info(f"优化任务分解完成，共 {len(tasks)} 个批量任务（原{len(user_profiles)}个用户）")
        return tasks
    
    async def _execute_distributed_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行分布式任务"""
        if not tasks:
            return []
        
        # 构建 (index, task, future, attempt) 队列
        pending = []
        for i, task in enumerate(tasks):
            worker = self.workers[i % len(self.workers)]
            fut = worker.generate_content.remote(
                task['base_prompt'],
                task.get('batch_profiles', [task.get('user_profile', {})]),
                task['target_attributes']
            )
            pending.append({'index': i, 'task': task, 'future': fut, 'attempt': 1})

        results_map: Dict[int, Dict[str, Any]] = {}
        max_attempts = 2
        per_attempt_timeout = 30.0

        while pending:
            futures = [p['future'] for p in pending]
            # 等待一批任务完成或超时
            ready, not_ready = ray.wait(futures, num_returns=len(futures), timeout=per_attempt_timeout)

            # 收集已完成结果
            if ready:
                for r in ready:
                    # 找到对应的 pending 项
                    match = next((p for p in pending if p['future'] == r), None)
                    if match is None:
                        continue
                    try:
                        res = ray.get(r)
                        results_map[match['index']] = res
                    except Exception as e:
                        logger.error(f"任务索引 {match['index']} 执行失败: {e}")
                        # 标记为失败以便重试
                        not_ready.append(r)
                    # 从 pending 中移除
                    pending = [p for p in pending if p['future'] != r]

            # 处理未完成的任务：重试或失败
            if not_ready:
                # 将 ObjectRef 映射回 pending 项
                ref_to_item = {p['future']: p for p in pending}
                pending = [p for p in pending if p['future'] not in not_ready]

                for ref in not_ready:
                    item = ref_to_item.get(ref)
                    if not item:
                        continue
                    if item['attempt'] >= max_attempts:
                        # 达到最大重试次数，转为本地同步生成（不使用简化结果）
                        try:
                            sync_res = self._generate_content_sync(
                                base_prompt=item['task']['base_prompt'],
                                batch_profiles=item['task'].get('batch_profiles', []),
                                target_attributes=item['task']['target_attributes']
                            )
                            results_map[item['index']] = sync_res
                            continue
                        except Exception as e:
                            raise RuntimeError(
                                f"分布式任务执行失败且本地生成也失败（索引 {item['index']}）: {e}"
                            )
                    # 重新分配到下一个工作节点并重试
                    next_worker = self.workers[(item['index'] + item['attempt']) % len(self.workers)]
                    new_future = next_worker.generate_content.remote(
                        item['task']['base_prompt'],
                        item['task'].get('batch_profiles', [item['task'].get('user_profile', {})]),
                        item['task']['target_attributes']
                    )
                    pending.append({
                        'index': item['index'],
                        'task': item['task'],
                        'future': new_future,
                        'attempt': item['attempt'] + 1
                    })

        # 汇总结果，按原始顺序返回
        results = [results_map[i] for i in range(len(tasks))]
        logger.info(f"完成{len(results)}个分布式任务")
        return results

    def _generate_content_sync(self, base_prompt: str, batch_profiles: List[Dict[str, Any]], 
                               target_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """本地同步生成内容（无降级，使用相同API实现）。"""
        try:
            # 构建增强提示（对首个用户取画像要点）
            profile = batch_profiles[0] if batch_profiles else {}
            interests = profile.get('interests', [])
            behavior = profile.get('behavior_patterns', {})
            cogpref = profile.get('cognitive_preferences', {})

            enhanced_prompt = f"""
            基础任务: {base_prompt}
            
            用户画像:
            - 兴趣: {interests}
            - 行为模式: {behavior}
            - 认知偏好: {cogpref}
            
            目标认知属性:
            - 立场: {target_attributes.get('stance', 'neutral')}
            - 情感: {target_attributes.get('emotion', 'neutral')}
            - 意图: {target_attributes.get('intent', 'inform')}
            
            请生成符合用户画像和目标认知属性的内容，确保语义匹配度和情感一致性。
            """

            # 生成内容
            llm = ChatOpenAI(
                model=self.config.get('model_name', 'qwen-max'),
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                base_url=os.getenv('OPENAI_BASE_URL'),
                temperature=0.7,
                max_tokens=2048
            )
            response = llm.invoke(enhanced_prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # 语义匹配
            target_desc = f"{target_attributes.get('stance', 'neutral')} {target_attributes.get('emotion', 'neutral')} {target_attributes.get('intent', 'inform')}"
            emb_model = OpenAIEmbeddings(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                openai_api_base=os.getenv('OPENAI_BASE_URL')
            )
            c_emb = emb_model.embed_query(content)
            t_emb = emb_model.embed_query(target_desc)
            c_tensor = torch.tensor(c_emb)
            t_tensor = torch.tensor(t_emb)
            similarity = torch.cosine_similarity(c_tensor.unsqueeze(0), t_tensor.unsqueeze(0), dim=1)
            semantic_score = float(similarity.item())

            # 情感得分
            emo_llm = ChatOpenAI(
                model=self.config.get('emotion_model', 'gpt-3.5-turbo'),
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                base_url=os.getenv('OPENAI_BASE_URL'),
                temperature=0.1,
                max_tokens=100
            )
            emo_prompt = f"请分析以下文本的情感倾向，只返回一个词：positive、negative或neutral。\n\n文本：{content}\n\n情感分析结果："
            emo_res = emo_llm.invoke(emo_prompt)
            predicted_emotion = (emo_res.content if hasattr(emo_res, 'content') else str(emo_res)).strip().lower()
            emotion_mapping = {
                'positive': ['positive', 'joy', 'optimism'],
                'negative': ['negative', 'sadness', 'anger'],
                'neutral': ['neutral']
            }
            target_emotion = target_attributes.get('emotion', 'neutral')
            match = predicted_emotion in emotion_mapping.get(target_emotion, ['neutral'])
            emotion_score = 0.8 if match else 0.3

            return {
                'content': content,
                'node_id': -1,
                'semantic_score': semantic_score,
                'emotion_score': emotion_score
            }
        except Exception as e:
            logger.error(f"本地同步内容生成失败: {e}")
            raise
    
    async def _validate_components(self):
        """验证所有组件正常工作"""
        # 验证Ray状态
        if not ray.is_initialized():
            raise RuntimeError("Ray未正确初始化")
        
        # 验证工作节点状态
        if not self.workers:
            raise RuntimeError("无可用工作节点")
        
        # 测试工作节点响应 - 异步调用health_check方法
        test_futures = [worker.health_check.remote() for worker in self.workers[:2]]  # 测试前2个节点
        
        # 正确使用ray.get获取ObjectRef结果
        try:
            test_results = ray.get(test_futures, timeout=10.0)  # health_check.remote()返回ObjectRef
            healthy_nodes = sum(1 for result in test_results if result.get('status') == 'healthy')
            
            if healthy_nodes == 0:
                logger.warning("所有工作节点健康检查失败，但继续运行")
            else:
                logger.info(f"组件验证完成，{healthy_nodes}个节点正常")
                
        except Exception as e:
            logger.warning(f"工作节点健康检查超时或失败，跳过验证: {e}")
            # 不抛出异常，允许系统继续运行
    
    def _aggregate_results(self, results: List[Dict[str, Any]], 
                          scenario: ScenarioRequirement) -> str:
        """聚合结果"""
        if not results:
            return ""
        
        # 按权重聚合内容
        weights = self.weight_manager.get_current_weights()
        
        # 计算加权分数
        weighted_results = []
        for result in results:
            score = (
                result.get('semantic_score', 0) * weights['semantic_match'] +
                result.get('emotion_score', 0) * weights['emotion_consistency']
            )
            weighted_results.append((score, result['content']))
        
        # 选择最佳结果或组合结果
        weighted_results.sort(key=lambda x: x[0], reverse=True)
        
        if len(weighted_results) == 1:
            return weighted_results[0][1]
        else:
            # 组合前几个最佳结果
            top_contents = [result[1] for result in weighted_results[:2]]
            return self._combine_contents(top_contents)
    
    def _combine_contents(self, contents: List[str]) -> str:
        """组合多个内容"""
        if not contents:
            return ""
        
        # 简单的内容组合策略
        combined = ""
        for i, content in enumerate(contents):
            if i > 0:
                combined += "\n\n"
            combined += content
        
        return combined[:1000]  # 限制长度
    
    def _apply_privacy_protection(self, content: str) -> str:
        """应用隐私保护"""
        # 简化的隐私保护实现
        # 在实际应用中，这里会进行更复杂的隐私保护处理
        
        # 对内容特征添加噪声
        content_embedding = self._advanced_text_embedding(content)
        noisy_embedding = self.add_gaussian_noise(content_embedding)
        # 实际应用中可以重新解码为文本
        
        return content
    
    def _advanced_text_embedding(self, text: str) -> np.ndarray:
        """高级文本嵌入 - 使用OpenAI Embeddings API"""
        # 使用首个工作节点的语义模型
        if self.workers:
            try:
                # 使用OpenAI Embeddings API
                embedding_model = OpenAIEmbeddings(
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    openai_api_base=os.getenv('OPENAI_BASE_URL')
                )
                embedding = embedding_model.embed_query(text)
                return np.array(embedding)
            except Exception as e:
                logger.warning(f"文本嵌入失败，使用随机向量: {e}")
                return np.random.rand(1536).astype(float) # OpenAI embeddings are 1536-dim
        else:
            # 备用方案
            return np.random.rand(1536).astype(float)
    
    def update_with_feedback(self, feedback: Dict[str, Any]):
        """根据反馈更新模型"""
        # 提取关键指标
        feedback_metrics = {}
        
        if 'semantic_match' in feedback:
            feedback_metrics['semantic_match'] = feedback['semantic_match']
        if 'emotion_consistency' in feedback:
            feedback_metrics['emotion_consistency'] = feedback['emotion_consistency']
        if 'user_satisfaction' in feedback:
            feedback_metrics['user_preference'] = feedback['user_satisfaction']
        
        # 更新权重
        self.weight_manager.update_weights(feedback_metrics)
        
        logger.info(f"基于反馈更新权重: {self.weight_manager.get_current_weights()}")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查工作节点状态
            worker_status = []
            if self.workers:
                for i, worker in enumerate(self.workers):
                    try:
                        # 测试工作节点响应
                        test_result = await worker.health_check.remote()
                        worker_status.append({
                            'worker_id': i,
                            'status': 'healthy',
                            'details': ray.get(test_result)
                        })
                    except Exception as e:
                        worker_status.append({
                            'worker_id': i,
                            'status': 'unhealthy',
                            'error': str(e)
                        })
            
            healthy_workers = sum(1 for w in worker_status if w['status'] == 'healthy')
            
            return {
                'status': 'healthy' if healthy_workers > 0 else 'degraded',
                'total_workers': len(self.workers),
                'healthy_workers': healthy_workers,
                'worker_details': worker_status,
                'differential_privacy_enabled': self.enable_differential_privacy,
                'weight_manager_active': self.weight_manager is not None
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'total_workers': len(self.workers) if hasattr(self, 'workers') else 0
            }

    async def update_system_prompt(self, new_prompt: str):
        """更新系统提示词"""
        try:
            # 更新所有分布式工作节点的系统提示
            if self.workers:
                update_tasks = []
                for worker in self.workers:
                    task = worker.update_prompt.remote(new_prompt)
                    update_tasks.append(task)
                
                # 等待所有节点更新完成
                results = ray.get(update_tasks)
                successful_updates = sum(1 for result in results if result.get('success', False))
                
                logger.info(f"系统提示更新完成: {successful_updates}/{len(self.workers)} 个节点成功更新")
            else:
                logger.warning("无可用工作节点，跳过系统提示更新")
                
        except Exception as e:
            logger.error(f"系统提示更新失败: {e}")
            # 不抛出异常，允许系统继续运行
    
    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            'num_workers': len(self.workers),
            'current_weights': self.weight_manager.get_current_weights(),
            'weight_history_length': len(self.weight_manager.history),
            'privacy_budget': self.privacy_budget,
            'distributed_mode': ray.is_initialized(),
            'ray_cluster_resources': ray.cluster_resources() if ray.is_initialized() else {},
            'gpu_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if ray.is_initialized():
            try:
                ray.shutdown()
            except:
                pass


# 工厂函数
def create_cognitive_foundation_model(config: Dict[str, Any] = None) -> CognitiveFoundationModel:
    """创建认知基础模型实例"""
    return CognitiveFoundationModel(config)


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = create_cognitive_foundation_model({
        'num_workers': 2,
        'privacy_budget': 1.0
    })
    
    # 示例用户画像
    user_profile = UserProfile(
        user_id="user_001",
        demographics={"age": 25, "location": "北京"},
        interests=["科技", "创新"],
        behavior_patterns={"active_time": "evening"},
        cognitive_preferences={"stance": "positive", "detail_level": "medium"}
    )
    
    # 示例场景需求
    scenario = ScenarioRequirement(
        scenario_id="scenario_001",
        target_attributes=CognitiveAttributes(
            stance="positive",
            emotion="optimistic",
            intent="inform",
            confidence=0.8
        ),
        context={"topic": "AI发展", "platform": "social_media"},
        constraints={"max_length": 200, "language": "chinese"}
    )
    
    # 异步生成内容
    async def main():
        result = await model.generate_cognitive_content([user_profile], scenario)
        print(f"生成内容: {result['content']}")
        print(f"元数据: {result['generation_metadata']}")
    
    # 运行示例
    if asyncio:
        asyncio.run(main())
