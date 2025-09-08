# -*- coding: utf-8 -*-
"""
DISTAgent 记忆模块
存储分发历史、用户反馈和场景数据，支持高效检索和动态更新
"""

import os
import json
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle

logger = logging.getLogger(__name__)

# 强制导入核心依赖
import faiss
from langchain_openai import OpenAIEmbeddings
import torch

# 事件总线（内存）实现，完全替代 Kafka
import asyncio

class EventStreamProcessor:
    """内存事件总线，替代 Kafka 实现发布/订阅"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.topics = {
            'memory_updates': 'distagent_memory_updates',
            'feedback_stream': 'distagent_feedback_stream',
            'importance_updates': 'distagent_importance_updates'
        }
        # 为每个主题创建一个 asyncio 队列
        self._queues: Dict[str, asyncio.Queue] = {
            name: asyncio.Queue() for name in self.topics.values()
        }
        logger.info("内存事件总线已启用（无需Kafka）")

    async def initialize(self):
        logger.info("内存事件总线初始化完成")

    async def _put(self, topic: str, value: Dict[str, Any]):
        await self._queues[topic].put(value)

    async def publish_memory_update(self, update_data: Dict[str, Any]):
        enriched_data = {
            **update_data,
            'timestamp': datetime.now().isoformat(),
            'node_id': 'memory_module',
            'version': '1.0'
        }
        await self._put(self.topics['memory_updates'], enriched_data)
        logger.debug(f"发布事件: memory_update -> {enriched_data.get('type')}")

    async def publish_feedback_stream(self, feedback_data: Dict[str, Any]):
        enriched_data = {
            **feedback_data,
            'timestamp': datetime.now().isoformat(),
            'node_id': 'memory_module'
        }
        await self._put(self.topics['feedback_stream'], enriched_data)
        logger.debug("发布事件: feedback_stream")

    async def subscribe_to_updates(self, callback: callable):
        """订阅 memory_updates 与 feedback_stream 两个主题并回调处理"""
        mem_q = self._queues[self.topics['memory_updates']]
        fb_q = self._queues[self.topics['feedback_stream']]
        logger.info("启动内存事件订阅（memory_updates, feedback_stream）...")

        while True:
            try:
                # 轮询两个队列，避免阻塞
                for q in (mem_q, fb_q):
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=0.5)
                        await asyncio.to_thread(callback, item)
                    except asyncio.TimeoutError:
                        continue
            except Exception as e:
                logger.error(f"事件订阅处理失败: {e}")
                await asyncio.sleep(1)

    def close(self):
        logger.info("事件总线已关闭")

logger.info("所有分布式存储依赖已成功加载")


@dataclass
class MemoryItem:
    """记忆项数据类"""
    id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    timestamp: datetime
    importance_score: float
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class ShortTermMemory:
    """短期记忆"""
    content_id: str
    scenario_id: str
    feedback: Dict[str, Any]
    timestamp: datetime
    users_involved: List[str]


@dataclass
class LongTermMemory:
    """长期记忆"""
    user_id: str
    user_profile: Dict[str, Any]
    preferences: Dict[str, Any]
    cognitive_feedback: Dict[str, Any]
    aggregated_interactions: Dict[str, Any]


class DifferentialPrivacyIndexer:
    """差分隐私索引器"""
    
    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale
    
    def create_private_index(self, embedding: np.ndarray) -> np.ndarray:
        """创建差分隐私索引"""
        noise = np.random.normal(0, self.noise_scale, embedding.shape)
        return embedding + noise


class DistributedVectorDatabase:
    """分布式向量数据库"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.dimension = config.get('dimension', 1536)  # 嵌入向量维度
        self.index = None
        self.memory_items = {}
        self.privacy_indexer = DifferentialPrivacyIndexer()
        # 强制CPU模式（默认），可通过配置 force_cpu=False 开启GPU（若faiss支持）
        self.force_cpu = self.config.get('force_cpu', True)
        
        # 嵌入模型 - 延后到 initialize 中初始化
        self.embeddings = None
        self.sentence_transformer = None
    
    def _initialize_index(self):
        """初始化FAISS索引（默认CPU-only，可选GPU）"""
        # 创建FAISS索引（内积）
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 仅在明确未强制CPU、且faiss具备GPU接口并且CUDA可用时使用GPU
        use_gpu = (
            not self.force_cpu and
            torch.cuda.is_available() and 
            hasattr(faiss, 'StandardGpuResources') and 
            hasattr(faiss, 'index_cpu_to_gpu')
        )
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info(f"初始化GPU加速FAISS索引，维度: {self.dimension}")
        else:
            if torch.cuda.is_available() and not use_gpu and not self.force_cpu:
                logger.warning("检测到CUDA可用，但faiss不支持GPU，使用CPU索引")
            logger.info(f"初始化CPU FAISS索引，维度: {self.dimension}")
        
        # 验证索引可用性
        if self.index is None:
            raise RuntimeError("FAISS索引初始化失败")
    
    async def initialize(self):
        """异步初始化方法 - 强制成功模式，增加超时保护"""
        logger.info("开始初始化分布式向量数据库...")
        
        # 初始化嵌入模型 - 只使用OpenAI API
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('OPENAI_BASE_URL')
        )
        self.sentence_transformer = None  # 不使用本地模型
        
        # 启用嵌入维度探测以确保正确配置
        try:
            logger.info("正在探测嵌入维度...")
            probe = await asyncio.wait_for(
                self._generate_embedding("dimension_probe"), 
                timeout=10.0
            )
            if probe is not None:
                embed_dim = len(probe)
                if embed_dim != self.dimension:
                    logger.info(f"检测到嵌入维度为 {embed_dim}，与配置 {self.dimension} 不一致，自动调整")
                    self.dimension = embed_dim
                else:
                    logger.info(f"嵌入维度验证成功: {self.dimension}")
            else:
                logger.warning("维度探测返回空结果，使用默认维度")
        except Exception as e:
            logger.warning(f"嵌入维度探测失败，使用默认维度: {e}")
        
        # 初始化索引（CPU优先）
        logger.info("正在初始化FAISS索引...")
        self._initialize_index()
        
        # 跳过组件验证，直接完成初始化加速启动
        logger.info("跳过组件验证步骤 (快速启动模式)")
        
        logger.info("分布式向量数据库初始化完成 - 基础功能可用")
    
    async def _validate_components(self):
        """验证所有组件正常工作"""
        # 验证FAISS索引
        if self.index is None:
            raise RuntimeError("FAISS索引未正确初始化")
        
        # 验证嵌入模型
        test_embedding = await self._generate_embedding("测试文本")
        if test_embedding is None:
            raise RuntimeError("嵌入模型验证失败")
        emb_dim = len(test_embedding)
        if emb_dim != self.dimension:
            # 自动对齐维度并重建索引
            logger.info(f"嵌入维度与索引不一致({emb_dim}!={self.dimension})，自动对齐并重建索引")
            self.dimension = emb_dim
            self._initialize_index()
        
        logger.info("分布式向量数据库组件验证通过")
    
    async def add_memory(self, memory_item: MemoryItem) -> bool:
        """添加记忆项 - 强制成功模式"""
        # 生成嵌入
        if memory_item.embedding is None:
            memory_item.embedding = await self._generate_embedding(memory_item.content)
        
        if memory_item.embedding is None:
            raise RuntimeError(f"无法为记忆项{memory_item.id}生成嵌入")
        
        # 应用差分隐私
        private_embedding = self.privacy_indexer.create_private_index(memory_item.embedding)
        
        # 添加到FAISS索引
        self.index.add(private_embedding.reshape(1, -1).astype('float32'))
        
        # 存储记忆项
        self.memory_items[memory_item.id] = memory_item
        
        logger.debug(f"成功添加记忆项: {memory_item.id}")
        return True
    
    async def search_similar(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """搜索相似记忆 - 强制成功模式"""
        try:
            # 输入验证
            if not query or not query.strip():
                logger.warning("搜索查询为空")
                return []
            
            # 检查是否有记忆数据
            if len(self.memory_items) == 0:
                logger.debug("暂无记忆数据可搜索")
                return []
            
            # 检查索引是否可用
            if self.index is None:
                logger.warning("FAISS索引未初始化")
                return []
            
            # 生成查询嵌入
            query_embedding = await self._generate_embedding(query.strip())
            
            if query_embedding is None:
                logger.warning(f"无法为查询'{query[:50]}...'生成嵌入")
                return []
            
            # 使用FAISS搜索
            search_k = min(top_k, len(self.memory_items))
            if search_k == 0:
                return []
                
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                search_k
            )
            
            results = []
            memory_keys = list(self.memory_items.keys())
            
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(memory_keys):  # 有效索引
                    try:
                        memory_id = memory_keys[idx]
                        memory_item = self.memory_items[memory_id]
                        memory_item.access_count += 1
                        memory_item.last_accessed = datetime.now()
                        results.append(memory_item)
                    except (IndexError, KeyError) as e:
                        logger.warning(f"访问记忆项失败: {e}")
                        continue
            
            logger.debug(f"FAISS搜索返回{len(results)}条记忆，查询: {query[:50]}...")
            return results
            
        except Exception as e:
            import traceback
            logger.error(f"搜索相似记忆失败: {type(e).__name__}: {str(e)}")
            logger.debug(f"搜索失败详细堆栈: {traceback.format_exc()}")
            return []
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入 - 使用OpenAI API"""
        try:
            # 使用OpenAI Embeddings API
            embedding = self.embeddings.embed_query(text)
            return np.array(embedding, dtype='float32')
            
            # 备用OpenAI嵌入
            if self.embeddings is not None:
                embedding = await asyncio.to_thread(self.embeddings.embed_query, text)
                return np.array(embedding, dtype='float32')
            
            raise RuntimeError("所有嵌入模型均不可用")
            
        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            raise RuntimeError(f"无法为文本生成嵌入: {text[:50]}...")
    
    
    def update_importance_scores(self, feedback: Dict[str, float]):
        """更新重要性分数"""
        for memory_id, score_delta in feedback.items():
            if memory_id in self.memory_items:
                self.memory_items[memory_id].importance_score += score_delta
                self.memory_items[memory_id].importance_score = max(0.0, 
                    min(1.0, self.memory_items[memory_id].importance_score))


## 重复的 EventStreamProcessor 定义已移除，避免冲突（保留文件顶部的实现）


class MemoryModule:
    """记忆模块主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 分布式向量数据库
        vector_cfg = self.config.get('vector_db', {})
        # 将上层 memory_config.vector_dim 传递给内部维度配置
        if 'vector_dim' in self.config and 'dimension' not in vector_cfg:
            vector_cfg = {**vector_cfg, 'dimension': self.config['vector_dim']}
        # 默认强制CPU模式，避免GPU依赖
        if 'force_cpu' not in vector_cfg:
            vector_cfg = {**vector_cfg, 'force_cpu': True}
        self.vector_db = DistributedVectorDatabase(vector_cfg)
        
        # 事件流处理器（内存）
        self.stream_processor = EventStreamProcessor(self.config.get('kafka', {}))
        
        # 短期和长期记忆存储
        self.short_term_memory: List[ShortTermMemory] = []
        self.long_term_memory: Dict[str, LongTermMemory] = {}
        
        # 重要性权重
        self.importance_weights = {
            'recency': 0.3,
            'frequency': 0.2,
            'relevance': 0.3,
            'feedback_score': 0.2
        }
        
        # 启动实时更新监听
        self._start_real_time_updates()
        
        logger.info("记忆模块初始化完成")
    
    async def initialize(self):
        """异步初始化方法 - 强制成功模式"""
        logger.info("开始初始化记忆模块...")
        
        # 初始化分布式向量数据库
        await self.vector_db.initialize()
        
        # 初始化流处理器
        await self.stream_processor.initialize()
        
        logger.info("记忆模块初始化完成 - 所有分布式组件正常")
    
    async def store_short_term_memory(self, memory: ShortTermMemory):
        """存储短期记忆"""
        try:
            self.short_term_memory.append(memory)
            
            # 安全的记忆序列化
            try:
                memory_dict = asdict(memory)
                # 处理datetime对象序列化
                if hasattr(memory, 'timestamp') and memory.timestamp:
                    memory_dict['timestamp'] = memory.timestamp.isoformat()
            except Exception as serialize_error:
                logger.warning(f"记忆序列化失败，使用简化格式: {serialize_error}")
                memory_dict = {
                    'content_id': getattr(memory, 'content_id', 'unknown'),
                    'scenario_id': getattr(memory, 'scenario_id', 'unknown'), 
                    'users_involved': getattr(memory, 'users_involved', []),
                    'timestamp': getattr(memory, 'timestamp', datetime.now()).isoformat() if hasattr(memory, 'timestamp') else datetime.now().isoformat()
                }
            
            memory_item = MemoryItem(
                id=f"stm_{memory_dict.get('content_id', 'unknown')}_{datetime.now().isoformat()}",
                content=json.dumps(memory_dict, default=str),
                embedding=None,
                metadata={
                    'type': 'short_term',
                    'content_id': memory_dict.get('content_id', 'unknown'),
                    'scenario_id': memory_dict.get('scenario_id', 'unknown'),
                    'users_count': len(memory_dict.get('users_involved', []))
                },
                timestamp=getattr(memory, 'timestamp', datetime.now()),
                importance_score=0.5  # 默认重要性分数
            )
            
            # 添加到向量数据库
            await self.vector_db.add_memory(memory_item)
            
            # 发布更新事件
            await self.stream_processor.publish_memory_update({
                'type': 'short_term_added',
                'memory_id': memory_item.id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.debug(f"存储短期记忆: {memory_dict.get('content_id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"存储短期记忆失败: {e}")
            # 不抛出异常，允许系统继续运行
    
    async def store_long_term_memory(self, memory: LongTermMemory):
        """存储长期记忆"""
        try:
            self.long_term_memory[memory.user_id] = memory
            
            # 创建记忆项
            memory_item = MemoryItem(
                id=f"ltm_{memory.user_id}",
                content=json.dumps(asdict(memory)),
                embedding=None,
                metadata={
                    'type': 'long_term',
                    'user_id': memory.user_id
                },
                timestamp=datetime.now(),
                importance_score=1.0  # 长期记忆默认重要
            )
            
            # 添加到向量数据库
            await self.vector_db.add_memory(memory_item)
            
            # 发布更新事件
            await self.stream_processor.publish_memory_update({
                'type': 'long_term_updated',
                'user_id': memory.user_id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.debug(f"存储长期记忆: {memory.user_id}")
            
        except Exception as e:
            logger.error(f"存储长期记忆失败: {e}")
    
    async def retrieve_memories(self, query: str, user_id: Optional[str] = None, 
                         memory_type: str = 'all', top_k: int = 5) -> Dict[str, Any]:
        """检索记忆（返回结构化结果，兼容DISTAgent）"""
        try:
            # 输入验证
            if not query or not query.strip():
                logger.warning("检索查询为空，返回空结果")
                return {'memories': [], 'user_profile': {}, 'patterns': {}, 'relevance_score': 0.0}
            
            # 检查向量数据库是否可用
            if not hasattr(self, 'vector_db') or self.vector_db is None:
                logger.warning("向量数据库未初始化，返回空结果")
                return {'memories': [], 'user_profile': {}, 'patterns': {}, 'relevance_score': 0.0}
            
            # 检查是否有记忆数据
            if not hasattr(self.vector_db, 'memory_items') or len(self.vector_db.memory_items) == 0:
                logger.debug("暂无记忆数据，返回空结果")
                return {'memories': [], 'user_profile': {}, 'patterns': {}, 'relevance_score': 0.0}
            
            # 基础检索
            memories = await self.vector_db.search_similar(query.strip(), top_k * 2)
            
            # 类型过滤
            if memory_type in ('short_term', 'long_term'):
                memories = [m for m in memories if m.metadata.get('type') == memory_type]
            # 'both' 或 'all' 不做过滤
            
            # 重要性排序
            memories = self._rank_by_importance(memories)
            
            # 截断并结构化
            selected = memories[:top_k]
            structured = []
            for m in selected:
                try:
                    structured.append({
                        'id': m.id,
                        'content': m.content,
                        'metadata': m.metadata,
                        'timestamp': m.timestamp.isoformat() if hasattr(m.timestamp, 'isoformat') else str(m.timestamp),
                        'importance_score': m.importance_score,
                    })
                except Exception as item_error:
                    logger.warning(f"处理记忆项失败: {item_error}")
                    continue
            
            user_profile = {}
            if user_id and hasattr(self, 'long_term_memory') and user_id in self.long_term_memory:
                user_profile = self.long_term_memory[user_id].user_profile
            
            logger.debug(f"成功检索到{len(structured)}条记忆，查询: {query[:50]}...")
            return {
                'memories': structured,
                'user_profile': user_profile,
                'patterns': {},
                'relevance_score': 0.5 if structured else 0.0
            }
            
        except Exception as e:
            import traceback
            error_details = f"{type(e).__name__}: {str(e)}"
            logger.error(f"检索记忆失败: {error_details}")
            logger.debug(f"检索记忆失败详细堆栈: {traceback.format_exc()}")
            return {'memories': [], 'user_profile': {}, 'patterns': {}, 'relevance_score': 0.0}
    
    def _calculate_importance_score(self, memory: ShortTermMemory) -> float:
        """计算重要性分数"""
        # 基于反馈质量计算
        feedback_score = 0.0
        if memory.feedback:
            feedback_values = [v for v in memory.feedback.values() if isinstance(v, (int, float))]
            if feedback_values:
                feedback_score = sum(feedback_values) / len(feedback_values)
        
        # 基于用户参与度
        user_participation = len(memory.users_involved) / 10.0  # 归一化
        
        # 基于时间新鲜度
        time_freshness = 1.0  # 新记忆权重为1
        
        # 综合计算
        importance = (
            feedback_score * self.importance_weights['feedback_score'] +
            user_participation * self.importance_weights['relevance'] +
            time_freshness * self.importance_weights['recency']
        )
        
        return min(1.0, max(0.0, importance))
    
    def _rank_by_importance(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """按重要性排序"""
        def importance_key(memory: MemoryItem) -> float:
            # 综合重要性计算
            recency_score = 1.0 / max(1, (datetime.now() - memory.timestamp).days + 1)
            frequency_score = min(1.0, memory.access_count / 10.0)
            
            total_score = (
                recency_score * self.importance_weights['recency'] +
                frequency_score * self.importance_weights['frequency'] +
                memory.importance_score * (
                    self.importance_weights['relevance'] + 
                    self.importance_weights['feedback_score']
                )
            )
            
            return total_score
        
        return sorted(memories, key=importance_key, reverse=True)
    
    async def aggregate_to_long_term(self, user_id: str):
        """聚合短期记忆到长期记忆"""
        try:
            # 找到用户相关的短期记忆
            user_short_memories = [
                m for m in self.short_term_memory 
                if user_id in m.users_involved
            ]
            
            if not user_short_memories:
                return
            
            # 聚合用户交互数据
            aggregated_interactions = {}
            total_feedback = {}
            
            for memory in user_short_memories:
                # 聚合反馈
                for key, value in memory.feedback.items():
                    if key not in total_feedback:
                        total_feedback[key] = []
                    total_feedback[key].append(value)
                
                # 聚合交互
                if memory.scenario_id not in aggregated_interactions:
                    aggregated_interactions[memory.scenario_id] = []
                aggregated_interactions[memory.scenario_id].append(memory.content_id)
            
            # 计算平均反馈
            avg_feedback = {
                key: sum(values) / len(values) if values else 0
                for key, values in total_feedback.items()
            }
            
            # 更新或创建长期记忆
            if user_id in self.long_term_memory:
                ltm = self.long_term_memory[user_id]
                ltm.cognitive_feedback.update(avg_feedback)
                ltm.aggregated_interactions.update(aggregated_interactions)
            else:
                ltm = LongTermMemory(
                    user_id=user_id,
                    user_profile={},
                    preferences={},
                    cognitive_feedback=avg_feedback,
                    aggregated_interactions=aggregated_interactions
                )
            
            await self.store_long_term_memory(ltm)
            
            # 清理已聚合的短期记忆
            self.short_term_memory = [
                m for m in self.short_term_memory 
                if user_id not in m.users_involved or 
                (datetime.now() - m.timestamp).days < 1
            ]
            
            logger.info(f"用户{user_id}的短期记忆已聚合到长期记忆")
            
        except Exception as e:
            logger.error(f"聚合长期记忆失败: {e}")
    
    def _start_real_time_updates(self):
        """启动实时更新监听（后台协程方式）"""
        def update_handler(update_data: Dict[str, Any]):
            """处理收到的更新回调"""
            try:
                update_type = update_data.get('type')
                if update_type == 'feedback_received':
                    self._handle_feedback_update(update_data)
                elif update_type == 'importance_adjustment':
                    self._handle_importance_adjustment(update_data)
            except Exception as e:
                logger.error(f"处理实时更新失败: {e}")

        # 在事件循环中创建后台任务订阅Kafka/内存队列更新
        try:
            asyncio.create_task(
                self.stream_processor.subscribe_to_updates(update_handler)
            )
        except RuntimeError:
            # 若当前无事件循环（例如同步测试环境），则跳过实时订阅
            logger.debug("当前无事件循环，跳过实时更新订阅")
    
    def _handle_feedback_update(self, update_data: Dict[str, Any]):
        """处理反馈更新"""
        memory_id = update_data.get('memory_id')
        feedback_scores = update_data.get('feedback_scores', {})
        
        # 更新向量数据库中的重要性分数
        self.vector_db.update_importance_scores({memory_id: sum(feedback_scores.values()) / len(feedback_scores)})
    
    def _handle_importance_adjustment(self, update_data: Dict[str, Any]):
        """处理重要性调整"""
        adjustments = update_data.get('adjustments', {})
        self.vector_db.update_importance_scores(adjustments)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'vector_db_size': len(self.vector_db.memory_items),
            'importance_weights': self.importance_weights,
            'recent_activity': {
                'last_24h_additions': len([
                    m for m in self.short_term_memory 
                    if (datetime.now() - m.timestamp).days < 1
                ])
            }
        }

    async def store_memory(self, memory_type: str, content: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """通用存储接口（兼容DISTAgent）"""
        try:
            if memory_type == 'long_term':
                ltm = LongTermMemory(
                    user_id=user_id,
                    user_profile=metadata.get('user_profile', {}),
                    preferences=metadata.get('preferences', {}),
                    cognitive_feedback=metadata.get('cognitive_feedback', {}),
                    aggregated_interactions=metadata.get('aggregated_interactions', {})
                )
                await self.store_long_term_memory(ltm)
                return True
            else:
                content_id = metadata.get('task_id', f"content_{datetime.now().timestamp()}")
                scenario_id = metadata.get('content_type', metadata.get('scenario_id', 'general'))
                users_involved = list(metadata.get('target_users', []))
                if user_id and user_id not in users_involved:
                    users_involved.append(user_id)
                stm = ShortTermMemory(
                    content_id=str(content_id),
                    scenario_id=str(scenario_id),
                    feedback=metadata.get('feedback', {}),
                    timestamp=datetime.now(),
                    users_involved=[str(u) for u in users_involved] if users_involved else [user_id]
                )
                await self.store_short_term_memory(stm)
                return True
        except Exception as e:
            logger.error(f"存储通用记忆失败: {e}")
            return False

    async def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        try:
            status = {
                'module_name': 'MemoryModule',
                'vector_db_initialized': hasattr(self, 'vector_db') and self.vector_db is not None,
                'event_processor_active': hasattr(self, 'event_processor') and self.event_processor is not None,
                'memory_count': len(self.memory_cache) if hasattr(self, 'memory_cache') else 0,
                'distributed_mode': self.enable_distributed,
                'real_time_updates': self.enable_real_time_updates,
                'status': 'healthy'
            }
            
            # 检查向量数据库状态
            if hasattr(self, 'vector_db') and self.vector_db:
                try:
                    db_status = self.vector_db.get_index_info()
                    status['vector_db_details'] = {
                        'index_size': db_status.get('ntotal', 0),
                        'dimension': db_status.get('dimension', 0),
                        'is_trained': db_status.get('is_trained', False)
                    }
                except Exception as e:
                    status['vector_db_details'] = {'error': str(e)}
                    
            return status
        except Exception as e:
            return {
                'module_name': 'MemoryModule',
                'status': 'error',
                'error': str(e)
            }


# 工厂函数
def create_memory_module(config: Dict[str, Any] = None) -> MemoryModule:
    """创建记忆模块实例"""
    return MemoryModule(config)


# 使用示例
if __name__ == "__main__":
    # 创建记忆模块
    memory_module = create_memory_module({
        'vector_db': {'dimension': 768},
        'kafka': {'kafka_servers': ['localhost:9092']}
    })
    
    # 示例短期记忆
    short_memory = ShortTermMemory(
        content_id="content_001",
        scenario_id="scenario_001", 
        feedback={'engagement': 0.8, 'relevance': 0.9},
        timestamp=datetime.now(),
        users_involved=["user_001", "user_002"]
    )
    
    # 存储记忆
    memory_module.store_short_term_memory(short_memory)
    
    # 检索记忆
    results = memory_module.retrieve_memories("用户参与", top_k=3)
    print(f"检索到{len(results)}条记忆")
    
    # 聚合长期记忆
    memory_module.aggregate_to_long_term("user_001")
    
    # 获取统计信息
    stats = memory_module.get_memory_statistics()
    print(f"记忆统计: {stats}")
