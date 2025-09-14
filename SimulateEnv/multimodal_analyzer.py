# -*- coding: utf-8 -*-
"""
多模态内容分析器

用于分析帖子中的图片和视频内容，为大模型提供多模态信息补充
"""

import re
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import aiohttp
from urllib.parse import urlparse
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class MultimodalAnalysisResult:
    """多模态分析结果"""
    post_id: str
    urls_processed: List[str]  # 成功处理的URL列表
    urls_failed: List[str]  # 失败的URL列表
    analysis: str  # 分析结果文本
    enhanced_content: str  # 增强后的内容（原文+分析）
    success: bool  # 是否成功
    error_message: Optional[str] = None


class MultimodalAnalyzer:
    """多模态内容分析器"""

    def __init__(self, model_name: str = "qwen-vl-max-2025-08-13", max_images: int = 5, timeout: int = 30,
                 use_cache: bool = True, cache_dir: str = None, cache_filename: str = None):
        """
        初始化多模态分析器

        Args:
            model_name: 多模态模型名称
            max_images: 每个帖子最多处理的图片数量
            timeout: 请求超时时间
            use_cache: 是否使用缓存
            cache_dir: 缓存目录路径
            cache_filename: 缓存文件名（不含扩展名），如果为None则使用默认文件名
        """
        # 加载环境变量
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config', '.env'))

        self.model_name = model_name
        self.max_images = max_images
        self.timeout = timeout
        self.use_cache = use_cache

        # 初始化缓存管理器
        if use_cache:
            try:
                from .multimodal_cache import MultimodalCache
                self.cache = MultimodalCache(cache_dir, cache_filename)
            except ImportError:
                logger.warning("无法导入MultimodalCache，禁用缓存功能")
                self.use_cache = False
                self.cache = None
        else:
            self.cache = None

        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )

    async def close(self):
        """关闭客户端连接"""
        try:
            await self.client.close()
        except:
            pass

    def parse_media_urls(self, img_urls_str=None, video_urls_str=None) -> Tuple[List[str], List[str]]:
        """
        从字符串中解析出媒体URL列表，分别返回图片和视频URL

        Args:
            img_urls_str: 图片URL字符串（可能是str、float/NaN等类型）
            video_urls_str: 视频URL字符串（可能是str、float/NaN等类型）

        Returns:
            (图片URL列表, 视频URL列表)
        """
        img_urls = []
        video_urls = []

        # 解析图片URL - 处理各种可能的数据类型
        img_str = self._safe_convert_to_string(img_urls_str)
        if img_str:
            img_urls = self._extract_urls_from_string(img_str)

        # 解析视频URL - 处理各种可能的数据类型
        video_str = self._safe_convert_to_string(video_urls_str)
        if video_str:
            video_urls = self._extract_urls_from_string(video_str)

        # 去重但保持分离，并限制图片数量
        img_urls = list(dict.fromkeys(img_urls))[:self.max_images]
        video_urls = list(dict.fromkeys(video_urls))

        logger.info(f"解析得到 {len(img_urls)} 个图片URL（限制{self.max_images}个），{len(video_urls)} 个视频URL")
        return img_urls, video_urls

    def _safe_convert_to_string(self, value) -> str:
        """
        安全地将各种类型的值转换为字符串，处理NaN、None、float、列表等情况

        Args:
            value: 待转换的值（可能是str、float、list、None等）

        Returns:
            转换后的字符串，如果无效则返回空字符串
        """
        if value is None:
            return ""

        # 处理列表类型 - 过滤掉无效元素并转换为JSON格式
        if isinstance(value, list):
            try:
                import pandas as pd
                # 过滤掉None、NaN、空字符串等无效元素
                valid_items = []
                for item in value:
                    if item is not None:
                        # 检查pandas NaN
                        try:
                            if not pd.isna(item):
                                str_item = str(item).strip()
                                if str_item and str_item.lower() != 'nan':
                                    valid_items.append(str_item)
                        except:
                            # 如果pandas检查失败，直接检查字符串
                            str_item = str(item).strip()
                            if str_item and str_item.lower() != 'nan':
                                valid_items.append(str_item)

                if valid_items:
                    # 转换为JSON格式字符串
                    import json
                    return json.dumps(valid_items, ensure_ascii=False)
                else:
                    return ""
            except Exception as e:
                # 如果处理失败，返回空字符串
                logger.debug(f"处理列表类型时出错: {e}")
                return ""

        # 处理pandas的NaN和numpy的NaN
        try:
            import pandas as pd
            if pd.isna(value):
                return ""
        except ImportError:
            pass

        # 处理float NaN
        if isinstance(value, float) and str(value).lower() == 'nan':
            return ""

        # 转换为字符串并去除首尾空格
        try:
            str_value = str(value).strip()
            return str_value if str_value and str_value.lower() != 'nan' else ""
        except:
            return ""

    def _extract_urls_from_string(self, url_string: str) -> List[str]:
        """
        从字符串中提取URL，支持多种格式：
        - JSON数组格式: ['url1', 'url2']
        - 逗号分隔: url1, url2
        - 分号分隔: url1; url2
        - 空格分隔: url1 url2
        """
        # 添加额外的安全检查
        if url_string is None:
            return []

        # 确保输入是字符串类型，如果不是则进行安全转换
        if not isinstance(url_string, str):
            url_string = self._safe_convert_to_string(url_string)

        if not url_string or not url_string.strip():
            return []

        urls = []
        url_string = url_string.strip()

        try:
            # 尝试JSON解析
            if url_string.startswith('[') and url_string.endswith(']'):
                parsed_list = json.loads(url_string)
                if isinstance(parsed_list, list):
                    # 清理URL，去除多余的引号和空格
                    cleaned_urls = []
                    for url in parsed_list:
                        if url:
                            clean_url = str(url).strip().strip("'\"")
                            if clean_url:
                                cleaned_urls.append(clean_url)
                    return cleaned_urls
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试各种分隔符
        for separator in [',', ';', '\n', '\t']:
            if separator in url_string:
                # 安全地处理分割后的每个部分
                split_parts = url_string.split(separator)
                candidate_urls = []
                for part in split_parts:
                    if part and isinstance(part, str):
                        clean_part = part.strip().strip("'\"")
                        if clean_part:
                            candidate_urls.append(clean_part)
                    elif part:  # 如果不是字符串但不为空
                        safe_part = self._safe_convert_to_string(part)
                        if safe_part:
                            candidate_urls.append(safe_part.strip("'\""))

                # 验证是否都是有效URL
                valid_urls = [url for url in candidate_urls if url and self._is_valid_url(url)]
                if len(valid_urls) > 1:  # 如果分割后有多个有效URL，使用这个结果
                    urls.extend(valid_urls)
                    return urls

        # 尝试用正则表达式提取URL
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]\']+(?:\.[^\s<>"{}|\\^`\[\]\']*)+'
        extracted_urls = re.findall(url_pattern, url_string)
        if extracted_urls:
            # 安全地清理提取的URL
            cleaned_urls = []
            for url in extracted_urls:
                if url and isinstance(url, str):
                    clean_url = url.strip().strip("'\"")
                    if clean_url:
                        cleaned_urls.append(clean_url)
                elif url:  # 如果不是字符串但不为空
                    safe_url = self._safe_convert_to_string(url)
                    if safe_url:
                        cleaned_urls.append(safe_url.strip("'\""))
            urls.extend([url for url in cleaned_urls if url])
            return urls

        # 最后，如果看起来像单个URL，直接添加
        clean_url = url_string.strip().strip("'\"")
        if self._is_valid_url(clean_url):
            urls.append(clean_url)

        return urls

    def _is_valid_url(self, url: str) -> bool:
        """检查是否为有效URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False

    async def _check_url_accessibility(self, url: str) -> bool:
        """
        检查URL是否可访问，使用更宽松的策略

        策略：
        1. 先尝试HEAD请求
        2. 如果HEAD失败，尝试GET请求前几个字节
        3. 接受更多的HTTP状态码（200, 206, 302, 301等）
        4. 设置合适的User-Agent和Headers
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control': 'no-cache',
            'Referer': url.split('/')[0] + '//' + url.split('/')[2] + '/' if len(url.split('/')) > 2 else url
        }

        connector = aiohttp.TCPConnector(ssl=False)  # 对于一些SSL证书问题更宽松
        timeout = aiohttp.ClientTimeout(total=10, connect=5)

        try:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            ) as session:

                # 先尝试HEAD请求
                try:
                    async with session.head(url) as response:
                        # 接受更多状态码：200 OK, 206 Partial Content, 301/302 Redirect等
                        if response.status in [200, 206, 301, 302, 304]:
                            return True
                        # 对于图片URL，有些服务器不支持HEAD，但GET可以
                        elif response.status in [405, 501]:  # Method Not Allowed, Not Implemented
                            pass  # 继续尝试GET
                        else:
                            return False
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass  # HEAD失败，尝试GET

                # 如果HEAD失败或不支持，尝试GET请求前1KB数据
                try:
                    headers_with_range = headers.copy()
                    headers_with_range['Range'] = 'bytes=0-1023'  # 只请求前1KB

                    async with session.get(url, headers=headers_with_range) as response:
                        # 接受200 OK和206 Partial Content
                        if response.status in [200, 206]:
                            # 简单检查是否是图片/视频内容
                            content_type = response.headers.get('content-type', '').lower()
                            if any(media_type in content_type for media_type in
                                   ['image/', 'video/', 'application/octet-stream']):
                                return True
                            # 即使content-type不明确，如果状态码正常也认为可访问
                            return True
                        elif response.status in [301, 302]:  # 重定向也算可访问
                            return True
                        else:
                            return False

                except (aiohttp.ClientError, asyncio.TimeoutError):
                    # 最后的fallback：对于一些特殊URL，直接认为可能可访问
                    # 特别是已知的图片托管域名
                    trusted_domains = [
                        'zhimg.com', 'qpic.cn', 'xhscdn.com', 'sinaimg.cn',
                        'amazonaws.com', 'cloudfront.net', 'jsdelivr.net'
                    ]

                    if any(domain in url.lower() for domain in trusted_domains):
                        logger.info(f"URL来自可信域名，跳过可访问性检查: {url}")
                        return True

                    return False

        except Exception as e:
            logger.debug(f"URL可访问性检查异常 {url}: {e}")

            # 最后的兜底策略：如果URL格式看起来合理，就尝试处理
            if self._is_valid_url(url) and any(keyword in url.lower() for keyword in
                                               ['jpg', 'jpeg', 'png', 'gif', 'webp', 'img', 'image', 'pic']):
                logger.info(f"URL格式合理，允许尝试处理: {url}")
                return True

            return False

    async def analyze_post_media(self, post_id: str, content: str,
                                 img_urls: str = None, video_urls: str = None) -> MultimodalAnalysisResult:
        """
        分析帖子的多媒体内容

        Args:
            post_id: 帖子ID
            content: 帖子原始文本内容
            img_urls: 图片URL字符串
            video_urls: 视频URL字符串

        Returns:
            分析结果
        """
        # 首先检查缓存（如果启用且不强制刷新）
        if self.use_cache and self.cache:
            cached_result = self.cache.get_cached_result(post_id)
            if cached_result:
                logger.info(f"✅ 从缓存获取多模态分析结果: {post_id}")

                # 构建增强内容
                enhanced_content = content
                if cached_result['analysis_text']:
                    enhanced_content = f"{content}\n\n[多媒体内容分析]: {cached_result['analysis_text']}"

                return MultimodalAnalysisResult(
                    post_id=post_id,
                    urls_processed=cached_result['processed_urls'],
                    urls_failed=cached_result['failed_urls'],
                    analysis=cached_result['analysis_text'],
                    enhanced_content=enhanced_content,
                    success=True,
                    error_message=""
                )

        # 解析URL，分别获取图片和视频URL
        img_url_list, video_url_list = self.parse_media_urls(img_urls, video_urls)

        if not img_url_list and not video_url_list:
            logger.info(f"帖子 {post_id} 没有可用的媒体URL")
            return MultimodalAnalysisResult(
                post_id=post_id,
                urls_processed=[],
                urls_failed=[],
                analysis="",
                enhanced_content=content,
                success=True,
                error_message="无媒体URL"
            )

        # 分别检查图片和视频URL的可访问性
        accessible_img_urls = []
        accessible_video_urls = []
        failed_urls = []

        # 检查图片URL
        for url in img_url_list:
            try:
                if await self._check_url_accessibility(url):
                    accessible_img_urls.append(url)
                else:
                    failed_urls.append(url)
                    logger.warning(f"图片URL不可访问: {url}")
            except Exception as e:
                failed_urls.append(url)
                logger.error(f"检查图片URL可访问性失败 {url}: {e}")

        # 检查视频URL
        for url in video_url_list:
            try:
                if await self._check_url_accessibility(url):
                    accessible_video_urls.append(url)
                else:
                    failed_urls.append(url)
                    logger.warning(f"视频URL不可访问: {url}")
            except Exception as e:
                failed_urls.append(url)
                logger.error(f"检查视频URL可访问性失败 {url}: {e}")

        if not accessible_img_urls and not accessible_video_urls:
            logger.warning(f"帖子 {post_id} 的所有URL都不可访问")
            return MultimodalAnalysisResult(
                post_id=post_id,
                urls_processed=[],
                urls_failed=failed_urls,
                analysis="",
                enhanced_content=content,
                success=True,
                error_message="所有URL都不可访问"
            )

        # 分别调用多模态模型分析
        analysis_parts = []
        processed_urls = []

        try:
            # 分析图片
            if accessible_img_urls:
                logger.info(f"分析 {len(accessible_img_urls)} 个图片URL")
                img_analysis = await self._call_multimodal_model_for_images(accessible_img_urls, content)
                if img_analysis:
                    analysis_parts.append(f"图片分析: {img_analysis}")
                processed_urls.extend(accessible_img_urls)

            # 分析视频（只选择第一个视频）
            if accessible_video_urls:
                selected_video_url = accessible_video_urls[0]  # 只选第一个视频
                logger.info(f"分析视频URL: {selected_video_url}")
                video_analysis = await self._call_multimodal_model_for_video(selected_video_url, content)
                if video_analysis:
                    analysis_parts.append(f"视频分析: {video_analysis}")
                processed_urls.append(selected_video_url)

            # 合并分析结果
            if analysis_parts:
                analysis = " | ".join(analysis_parts)
                enhanced_content = f"{content}\n\n[多媒体内容分析]: {analysis}"
            else:
                enhanced_content = content
                analysis = ""

            # 保存成功结果到缓存
            if self.use_cache and self.cache:
                try:
                    self.cache.save_result(
                        post_id=post_id,
                        analysis_text=analysis,
                        processed_urls=processed_urls,
                        failed_urls=failed_urls,
                        model_name=self.model_name
                    )
                    logger.info(f"💾 多模态分析结果已保存到缓存: {post_id}")
                except Exception as cache_e:
                    logger.warning(f"保存多模态分析结果到缓存时出错: {cache_e}")

            return MultimodalAnalysisResult(
                post_id=post_id,
                urls_processed=processed_urls,
                urls_failed=failed_urls,
                analysis=analysis,
                enhanced_content=enhanced_content,
                success=True
            )

        except Exception as e:
            logger.error(f"多模态分析失败 {post_id}: {e}")

            # 即使失败也保存到缓存（避免重复失败的请求）
            if self.use_cache and self.cache:
                try:
                    self.cache.save_result(
                        post_id=post_id,
                        analysis_text="",
                        processed_urls=[],
                        failed_urls=accessible_img_urls + accessible_video_urls + failed_urls,
                        model_name=self.model_name
                    )
                except Exception as cache_e:
                    logger.warning(f"保存失败结果到缓存时出错: {cache_e}")

            return MultimodalAnalysisResult(
                post_id=post_id,
                urls_processed=[],
                urls_failed=accessible_img_urls + accessible_video_urls + failed_urls,
                analysis="",
                enhanced_content=content,
                success=False,
                error_message=str(e)
            )

    async def _call_multimodal_model_for_images(self, image_urls: List[str], content: str) -> str:
        """
        调用多模态大模型分析图片内容

        Args:
            image_urls: 图片URL列表
            content: 帖子原始内容

        Returns:
            分析结果文本
        """
        try:
            # 构建消息内容
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""请分析以下帖子中的图片内容，并提供简洁的描述分析。

原帖内容: {content}

请分析这些图片内容，重点关注：
1. 图片的主要内容是什么
2. 可能传达的情感或态度
3. 任何值得注意的细节

请用简洁的中文回复，不超过150字。"""
                        }
                    ]
                }
            ]

            # 添加图片URL
            for img_url in image_urls:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })

            # 调用模型API
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # max_tokens=200,
                    # temperature=0.3
                ),
                timeout=self.timeout
            )

            analysis = response.choices[0].message.content.strip()
            # 去除多余符号，换行等
            analysis = re.sub(r'[\n\r]+', ' ', analysis).strip()
            logger.info(f"图片分析完成，生成 {len(analysis)} 字符的分析")
            return analysis

        except asyncio.TimeoutError:
            logger.error(f"图片分析超时 ({self.timeout}s)")
            raise Exception("图片分析超时")
        except Exception as e:
            logger.error(f"图片分析失败: {e}")
            raise e

    async def _call_multimodal_model_for_video(self, video_url: str, content: str) -> str:
        """
        调用多模态大模型分析视频内容

        Args:
            video_url: 视频URL
            content: 帖子原始内容

        Returns:
            分析结果文本
        """
        try:
            # 构建消息内容
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""请分析以下帖子中的视频内容，并提供简洁的描述分析。

原帖内容: {content}

请分析这个视频内容，重点关注：
1. 视频的主要内容是什么
2. 可能传达的情感或态度
3. 任何值得注意的细节

请用简洁的中文回复，不超过150字。"""
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url}
                        }
                    ]
                }
            ]

            # 调用模型API
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # max_tokens=200,
                    # temperature=0.3
                ),
                timeout=self.timeout
            )

            analysis = response.choices[0].message.content.strip()
            logger.info(f"视频分析完成，生成 {len(analysis)} 字符的分析")
            return analysis

        except asyncio.TimeoutError:
            logger.error(f"视频分析超时 ({self.timeout}s)")
            raise Exception("视频分析超时")
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            raise e

    async def batch_analyze_posts(self, posts_data: List[Dict[str, Any]]) -> Dict[str, MultimodalAnalysisResult]:
        """
        批量分析多个帖子的媒体内容

        Args:
            posts_data: 帖子数据列表，每个元素包含post_id, content, img_urls, video_urls

        Returns:
            {post_id: MultimodalAnalysisResult} 的字典
        """
        logger.info(f"开始批量分析 {len(posts_data)} 个帖子的媒体内容")

        tasks = []
        for post_data in posts_data:
            task = self.analyze_post_media(
                post_id=post_data.get('post_id', ''),
                content=post_data.get('content', ''),
                img_urls=post_data.get('img_urls'),
                video_urls=post_data.get('video_urls')
            )
            tasks.append(task)

        # 并发执行分析任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 组织结果
        analysis_results = {}
        success_count = 0
        error_count = 0

        for i, result in enumerate(results):
            post_data = posts_data[i]
            post_id = post_data.get('post_id', f'unknown_{i}')

            if isinstance(result, MultimodalAnalysisResult):
                analysis_results[post_id] = result
                if result.success:
                    success_count += 1
                else:
                    error_count += 1
            else:
                # 处理异常结果
                error_count += 1
                analysis_results[post_id] = MultimodalAnalysisResult(
                    post_id=post_id,
                    urls_processed=[],
                    urls_failed=[],
                    analysis="",
                    enhanced_content=post_data.get('content', ''),
                    success=False,
                    error_message=str(result)
                )

        logger.info(f"批量分析完成：成功 {success_count}，失败 {error_count}")
        return analysis_results
