"""
多模态缓存管理模块

管理多模态分析结果的存储和检索，避免重复分析相同的内容
"""

import os
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib


class MultimodalCache:
    """多模态分析结果缓存管理器"""

    def __init__(self, cache_dir: str = None, cache_filename: str = None):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
            cache_filename: 缓存文件名（不含扩展名），如果为None则使用默认文件名
        """
        if cache_dir is None:
            # 使用项目根目录下的Output/multimodal_cache文件夹
            project_root = os.path.dirname(os.path.dirname(__file__))
            cache_dir = os.path.join(project_root, 'Output', 'multimodal_cache')

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 设置缓存文件名
        if cache_filename is None:
            cache_filename = 'multimodal_results'

        # 缓存文件路径
        self.cache_file = self.cache_dir / f'{cache_filename}.csv'
        self.metadata_file = self.cache_dir / f'{cache_filename}_metadata.json'

        # 初始化缓存文件
        self._init_cache_files()

    def _init_cache_files(self):
        """初始化缓存文件"""
        # 初始化CSV文件（如果不存在）
        if not self.cache_file.exists():
            with open(self.cache_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'post_id',           # 帖子ID（主键）
                    'analysis_text',     # 多模态分析文本
                    'processed_urls',    # 处理成功的URL列表（JSON）
                    'failed_urls',       # 处理失败的URL列表（JSON）
                    'model_name',        # 使用的模型名称
                    'created_at',        # 创建时间
                    'updated_at'         # 更新时间
                ])

        # 初始化元数据文件（如果不存在）
        if not self.metadata_file.exists():
            metadata = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'total_entries': 0,
                'description': '多模态分析结果缓存'
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """计算内容的MD5哈希值"""
        if not content:
            return ""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _normalize_urls(self, urls_str) -> str:
        """标准化URL字符串，用于哈希计算，处理各种数据类型"""
        if urls_str is None:
            return ""

        # 处理pandas NaN
        try:
            import pandas as pd
            if pd.isna(urls_str):
                return ""
        except ImportError:
            pass

        # 处理float NaN
        if isinstance(urls_str, float):
            if str(urls_str).lower() == 'nan':
                return ""
            # 如果是普通float，转换为字符串
            urls_str = str(urls_str)

        # 处理列表类型
        if isinstance(urls_str, list):
            try:
                # 过滤无效元素并排序
                valid_urls = []
                for url in urls_str:
                    if url is not None:
                        try:
                            import pandas as pd
                            if not pd.isna(url):
                                str_url = str(url).strip()
                                if str_url and str_url.lower() != 'nan':
                                    valid_urls.append(str_url)
                        except:
                            str_url = str(url).strip()
                            if str_url and str_url.lower() != 'nan':
                                valid_urls.append(str_url)

                if valid_urls:
                    return json.dumps(sorted(valid_urls), sort_keys=True)
                else:
                    return ""
            except:
                return ""

        # 转换为字符串
        urls_str = str(urls_str).strip()
        if not urls_str or urls_str.lower() == 'nan':
            return ""

        try:
            # 尝试解析JSON格式的URL列表
            if urls_str.startswith('['):
                urls_list = json.loads(urls_str)
                if isinstance(urls_list, list):
                    # 排序并重新序列化，确保相同URL集合产生相同哈希
                    sorted_urls = sorted([str(url).strip() for url in urls_list if url])
                    return json.dumps(sorted_urls, sort_keys=True)
        except (json.JSONDecodeError, ValueError):
            pass

        # 如果不是JSON格式，直接返回去除空格的字符串
        return urls_str

    def get_cached_result(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的多模态分析结果

        Args:
            post_id: 帖子ID（主键）

        Returns:
            缓存的分析结果，如果不存在则返回None
        """
        if not self.cache_file.exists():
            return None

        try:
            df = pd.read_csv(self.cache_file)

            # 简单查找：只基于post_id匹配
            matches = df[df['post_id'] == post_id]
            if len(matches) > 0:
                # 返回最新的记录
                latest = matches.iloc[-1]

                # 解析JSON字段
                processed_urls = json.loads(latest['processed_urls']) if latest['processed_urls'] else []
                failed_urls = json.loads(latest['failed_urls']) if latest['failed_urls'] else []

                return {
                    'analysis_text': latest['analysis_text'],
                    'processed_urls': processed_urls,
                    'failed_urls': failed_urls,
                    'model_name': latest['model_name'],
                    'created_at': latest['created_at'],
                    'updated_at': latest['updated_at']
                }

        except Exception as e:
            print(f"⚠️ 读取多模态缓存时出错: {e}")

        return None

    def save_result(self, post_id: str, analysis_text: str = "", processed_urls: List[str] = None, failed_urls: List[str] = None, model_name: str = "") -> bool:
        """
        保存多模态分析结果到缓存

        Args:
            post_id: 帖子ID（主键）
            analysis_text: 分析文本
            processed_urls: 处理成功的URL列表
            failed_urls: 处理失败的URL列表
            model_name: 使用的模型名称

        Returns:
            是否保存成功
        """
        try:
            # 检查是否已存在相同记录（基于post_id）
            existing = self.get_cached_result(post_id)

            now = datetime.now().isoformat()

            # 准备数据行（简化版）
            row_data = [
                post_id,  # 只使用post_id作为主键
                analysis_text,
                json.dumps(processed_urls or [], ensure_ascii=False),
                json.dumps(failed_urls or [], ensure_ascii=False),
                model_name,
                existing['created_at'] if existing else now,  # 保留原创建时间
                now  # 更新时间
            ]

            # 追加到CSV文件
            with open(self.cache_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

            # 更新元数据
            self._update_metadata()

            return True

        except Exception as e:
            print(f"❌ 保存多模态缓存时出错: {e}")
            return False

    def _update_metadata(self):
        """更新缓存元数据"""
        try:
            if self.cache_file.exists():
                df = pd.read_csv(self.cache_file)
                total_entries = len(df)
            else:
                total_entries = 0

            metadata = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'total_entries': total_entries,
                'description': '多模态分析结果缓存',
                'last_updated': datetime.now().isoformat()
            }

            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️ 更新缓存元数据时出错: {e}")

    def clear_cache(self) -> bool:
        """清空缓存"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()

            self._init_cache_files()
            print("✅ 多模态缓存已清空")
            return True

        except Exception as e:
            print(f"❌ 清空缓存时出错: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'total_entries': 0,
            'unique_posts': 0,
            'cache_size_mb': 0,
            'oldest_entry': None,
            'newest_entry': None
        }

        try:
            if self.cache_file.exists():
                df = pd.read_csv(self.cache_file)
                stats['total_entries'] = len(df)
                stats['unique_posts'] = df['post_id'].nunique()

                # 计算文件大小
                stats['cache_size_mb'] = round(self.cache_file.stat().st_size / 1024 / 1024, 2)

                if len(df) > 0:
                    stats['oldest_entry'] = df['created_at'].min()
                    stats['newest_entry'] = df['updated_at'].max()

        except Exception as e:
            print(f"⚠️ 获取缓存统计信息时出错: {e}")

        return stats

    def list_cached_posts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        列出缓存的帖子

        Args:
            limit: 返回的最大记录数

        Returns:
            缓存的帖子列表
        """
        posts = []

        try:
            if self.cache_file.exists():
                df = pd.read_csv(self.cache_file)

                # 按更新时间排序，返回最新的记录
                df_sorted = df.sort_values('updated_at', ascending=False)

                for _, row in df_sorted.head(limit).iterrows():
                    posts.append({
                        'post_id': row['post_id'],
                        'analysis_preview': row['analysis_text'][:100] + '...' if len(row['analysis_text']) > 100 else row['analysis_text'],
                        'model_name': row['model_name'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    })

        except Exception as e:
            print(f"⚠️ 列出缓存帖子时出错: {e}")

        return posts
