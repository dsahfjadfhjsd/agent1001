"""
USPE2024数据整合脚本

处理USPE2024数据集，包括：
1. 将按多模态URL拆分的记录重新整合
2. 将英文的stance、sentiment、intent值映射为中文
3. 生成与XMSU7D数据集格式一致的统一数据格式
"""

import pandas as pd
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class USPE2024DataOrganizer:
    """USPE2024数据整合器"""

    def __init__(self, data_path: str):
        """
        初始化数据整合器

        Args:
            data_path: USPE2024数据目录路径
        """
        self.data_path = Path(data_path)

        # 英文到中文的映射字典
        self.stance_mapping = {
            'favor': '支持',
            'against': '反对',
            'neutral': '中立',
            'none': '中立'
        }

        self.sentiment_mapping = {
            'Positive': '积极',
            'Negative': '消极',
            'Neutral': '中立',
            'positive': '积极',
            'negative': '消极',
            'neutral': '中立'
        }

        self.intent_mapping = {
            'Campaign Mobilization': '竞选动员',
            'Support & Praise': '支持与赞扬',
            'Opposition/Confrontational': '反对/对抗',
            'Criticism/Advice': '批评/建议',
            'Policy Slogans': '政策口号',
            'Other': '其他',
            'Spam/Nonsensical': '垃圾/无意义',
            'Information Seeking': '信息寻求',
            'Emotional Expression': '情感表达',
            'Religious/Spiritual Support': '宗教/精神支持'
        }

        # 用于存储article_id到post_id的映射
        self.article_id_mapping = {}

    def generate_post_id(self, platform: str, article_id: str) -> str:
        """生成统一的post_id格式"""
        try:
            # 使用平台和原始ID生成6位哈希
            combined = f"{platform}_{article_id}"
            hash_object = hashlib.md5(combined.encode())
            hex_dig = hash_object.hexdigest()
            # 取前6位作为post_id
            post_id = f"post_{hex_dig[:6]}"
            return post_id
        except Exception as e:
            logger.warning(f"生成post_id失败: {e}")
            # 备用方案：使用平台前缀 + 哈希
            backup_hash = str(hash(combined))[-6:]
            return f"post_{backup_hash}"

    def convert_timestamp_to_date(self, timestamp_str: str) -> str:
        """将时间戳转换为日期格式(YYYY-MM-DD HH:mm)"""
        try:
            if pd.isna(timestamp_str) or timestamp_str == '' or timestamp_str == '0':
                return ''

            # 转换为整数时间戳
            timestamp = int(float(timestamp_str))

            # 转换为datetime对象
            dt = datetime.fromtimestamp(timestamp)

            # 返回YYYY-MM-DD HH:mm格式（精确到小时和分钟）
            return dt.strftime('%Y-%m-%d %H:%M')
        except Exception as e:
            logger.warning(f"时间戳转换失败: {timestamp_str}, 错误: {e}")
            return ''

    def safe_get_field(self, row: pd.Series, field: str, default='') -> str:
        """安全获取字段值"""
        try:
            if field in row.index:
                value = row[field]
                if pd.isna(value) or value == '' or value == 'nan':
                    return default
                return str(value).strip()
            return default
        except Exception as e:
            logger.warning(f"获取字段 {field} 时出错: {e}")
            return default

    def map_to_chinese(self, value: str, mapping_dict: Dict[str, str]) -> str:
        """将英文值映射为中文"""
        if not value or pd.isna(value):
            return ''

        # 尝试直接映射
        if value in mapping_dict:
            return mapping_dict[value]

        # 尝试小写映射
        value_lower = value.lower()
        for key, chinese_value in mapping_dict.items():
            if key.lower() == value_lower:
                return chinese_value

        # 如果没有找到映射，返回原值
        logger.warning(f"未找到映射: {value}")
        return value

    def consolidate_multimodal_urls(self, group_df: pd.DataFrame) -> Dict:
        """合并同一article_id的多模态URL"""
        # 收集所有图片URL
        img_urls = []
        video_urls = []

        for _, row in group_df.iterrows():
            img_url = self.safe_get_field(row, 'img_urls')
            video_url = self.safe_get_field(row, 'video_urls')

            if img_url:
                img_urls.append(img_url)
            if video_url:
                video_urls.append(video_url)

        # 去重并合并
        img_urls = list(set(img_urls))
        video_urls = list(set(video_urls))

        # 使用第一行的数据作为基础
        base_row = group_df.iloc[0]

        return {
            'img_urls': ','.join(img_urls) if img_urls else '',
            'video_urls': ','.join(video_urls) if video_urls else '',
            'base_row': base_row
        }

    def process_article_data(self, file_path: Path, platform: str) -> pd.DataFrame:
        """处理文章数据"""
        logger.info(f"处理文章数据: {file_path}")

        # 读取数据
        df = pd.read_csv(file_path)
        logger.info(f"原始数据: {len(df)} 条记录")

        # 按article_id分组并合并多模态URL
        grouped_data = []

        for article_id, group in df.groupby('article_id'):
            # 合并多模态URL
            consolidated = self.consolidate_multimodal_urls(group)
            base_row = consolidated['base_row']

            # 生成post_id
            post_id = self.generate_post_id(platform, str(article_id))
            self.article_id_mapping[str(article_id)] = post_id

            # 创建统一格式的记录
            article_data = {
                'platform': platform,
                'post_id': post_id,
                'original_article_id': str(article_id),
                'content': self.safe_get_field(base_row, 'content'),
                'created_time': self.safe_get_field(base_row, 'created_time'),
                'created_date': self.convert_timestamp_to_date(self.safe_get_field(base_row, 'created_time')),
                'like_count': self.safe_get_field(base_row, 'like_count', '0'),
                'comment_count': self.safe_get_field(base_row, 'comment_count', '0'),
                'share_count': self.safe_get_field(base_row, 'share_count', '0'),
                'uid': '',  # USPE2024数据中没有uid
                'username': self.safe_get_field(base_row, 'username'),
                'location': '',  # USPE2024数据中没有location
                'stance': self.map_to_chinese(self.safe_get_field(base_row, 'stance'), self.stance_mapping),
                'sentiment': self.map_to_chinese(self.safe_get_field(base_row, 'sentiment'), self.sentiment_mapping),
                'intent': self.map_to_chinese(self.safe_get_field(base_row, 'intent'), self.intent_mapping),
                'stance_content': self.safe_get_field(base_row, 'stance_content', ''),
                'sentiment_content': self.safe_get_field(base_row, 'sentiment_content'),
                'intent_content': self.safe_get_field(base_row, 'intent_content'),
                'video_urls': consolidated['video_urls'],
                'img_urls': consolidated['img_urls']
            }

            grouped_data.append(article_data)

        result_df = pd.DataFrame(grouped_data)
        logger.info(f"整合后数据: {len(result_df)} 条记录")

        return result_df

    def generate_realistic_comment_times(self, post_timestamp: str, num_comments: int,
                                         days_range: tuple = (0, 0)) -> List[str]:
        """
        为评论生成符合真实情况的时间分布

        Args:
            post_timestamp: 帖子发布时间戳
            num_comments: 需要生成时间的评论数量
            days_range: 评论分布的天数范围，默认3-5天

        Returns:
            生成的评论时间戳列表（已排序）
        """
        try:
            # 转换帖子时间戳
            post_time = datetime.fromtimestamp(int(float(post_timestamp)))

            # 随机选择总天数（3-5天之间）
            total_days = np.random.uniform(days_range[0], days_range[1])

            # 生成评论时间分布
            # 使用Beta分布来模拟真实的评论热度衰减
            # Beta(2, 5) 会产生前期多、后期少的分布
            # Beta(2, 3) 会产生峰值在前1/3左右的分布
            # 随机选择分布类型
            distribution_type = np.random.choice(['early_peak', 'middle_peak', 'gradual_decline', 'uniform'])
            # distribution_type = np.random.choice(['early_peak', 'middle_peak', 'gradual_decline'], p=[0.6, 0.3, 0.1])
            # 均匀分布
            # distribution_type = 'uniform'

            if distribution_type == 'early_peak':
                # 峰值在最开始，快速衰减
                beta_params = (2, 5)
            elif distribution_type == 'middle_peak':
                # 峰值在中间偏前
                beta_params = (2, 3)
            elif distribution_type == 'gradual_decline':
                # 逐渐衰减
                beta_params = (1.5, 4)
            elif distribution_type == 'uniform':
                # 均匀分布
                beta_params = (1, 1)

            # 生成归一化的时间点（0-1之间）
            time_points = np.random.beta(beta_params[0], beta_params[1], num_comments)

            # 映射到实际时间范围（单位：小时）
            total_hours = total_days * 24
            hours_offsets = time_points * total_hours

            # 添加一些随机性，使时间更自然
            # 添加小的随机扰动（±30分钟）
            random_minutes = np.random.uniform(-30, 30, num_comments)

            # 生成评论时间戳
            comment_times = []
            for hours, minutes in zip(hours_offsets, random_minutes):
                total_offset = timedelta(hours=hours, minutes=minutes)
                comment_time = post_time + total_offset
                # 转换为时间戳
                comment_timestamp = str(int(comment_time.timestamp()))
                comment_times.append(comment_timestamp)

            # 排序时间戳
            comment_times.sort()

            return comment_times

        except Exception as e:
            logger.warning(f"生成评论时间失败: {e}，将使用帖子原始时间")
            # 如果生成失败，返回帖子时间的列表
            return [post_timestamp] * num_comments

    def process_comment_data(self, file_path: Path, platform: str, articles_df: pd.DataFrame = None) -> pd.DataFrame:
        """处理评论数据"""
        logger.info(f"处理评论数据: {file_path}")

        # 读取数据
        df = pd.read_csv(file_path)
        logger.info(f"原始评论数据: {len(df)} 条记录")

        # 创建article_id到时间的映射，用于给评论赋予帖子时间
        article_time_mapping = {}
        if articles_df is not None:
            for _, article_row in articles_df.iterrows():
                article_id = str(article_row['original_article_id'])
                article_time_mapping[article_id] = {
                    'created_time': article_row['created_time'],
                    'created_date': article_row['created_date']
                }

        # 首先按article_id分组，统计每个帖子有多少评论没有时间
        comments_without_time = {}
        # 检查是否存在created_time字段
        has_created_time = 'created_time' in df.columns

        for article_id, group in df.groupby('article_id'):
            article_id_str = str(article_id)

            # 找出没有时间的评论
            if has_created_time:
                no_time_mask = group['created_time'].isna() | (group['created_time'] == '') | (group['created_time'] == '0')
                num_no_time = no_time_mask.sum()
            else:
                # 如果没有created_time字段，所有评论都需要生成时间
                num_no_time = len(group)

            if num_no_time > 0 and article_id_str in article_time_mapping:
                # 获取帖子时间
                post_timestamp = article_time_mapping[article_id_str]['created_time']

                # 生成符合真实分布的评论时间
                generated_times = self.generate_realistic_comment_times(
                    post_timestamp,
                    num_no_time
                )

                # 存储生成的时间
                comments_without_time[article_id_str] = {
                    'times': generated_times,
                    'index': 0  # 用于追踪已分配的时间索引
                }

                logger.info(f"为帖子 {article_id_str} 的 {num_no_time} 条评论生成了分布式时间")

        processed_data = []

        for _, row in df.iterrows():
            original_article_id = str(self.safe_get_field(row, 'article_id'))
            # 只有当字段存在时才获取created_time
            created_time = self.safe_get_field(row, 'created_time') if has_created_time else ''

            # 查找对应的post_id，如果没有找到则生成一个
            post_id = self.article_id_mapping.get(original_article_id)
            if not post_id:
                post_id = self.generate_post_id(platform, original_article_id)
                self.article_id_mapping[original_article_id] = post_id

            # 如果评论没有时间，使用预先生成的分布式时间
            if not created_time or created_time == '' or pd.isna(created_time):
                if original_article_id in comments_without_time:
                    # 获取下一个分配的时间
                    time_data = comments_without_time[original_article_id]
                    idx = time_data['index']
                    if idx < len(time_data['times']):
                        created_time = time_data['times'][idx]
                        created_date = self.convert_timestamp_to_date(created_time)
                        time_data['index'] += 1  # 移动到下一个时间
                        logger.debug(f"评论 {self.safe_get_field(row, 'comment_id')} 使用分布式时间: {created_date}")
                    else:
                        # 理论上不应该到这里，但作为保险
                        if original_article_id in article_time_mapping:
                            created_time = article_time_mapping[original_article_id]['created_time']
                            created_date = article_time_mapping[original_article_id]['created_date']
                        else:
                            created_time = ''
                            created_date = ''
                elif original_article_id in article_time_mapping:
                    # 如果没有在预生成列表中（理论上不应该发生），回退到使用帖子时间
                    created_time = article_time_mapping[original_article_id]['created_time']
                    created_date = article_time_mapping[original_article_id]['created_date']
                    logger.warning(f"评论 {self.safe_get_field(row, 'comment_id')} 未在预生成列表中，使用帖子时间")
                else:
                    created_time = ''
                    created_date = ''
                    # logger.warning(f"评论 {self.safe_get_field(row, 'comment_id')} 找不到对应帖子的时间，article_id: {original_article_id}")
            else:
                created_date = self.convert_timestamp_to_date(created_time)

            comment_data = {
                'platform': platform,
                'comment_id': self.safe_get_field(row, 'comment_id'),
                'content': self.safe_get_field(row, 'content'),
                'post_id': post_id,
                'original_article_id': original_article_id,
                'created_time': created_time,
                'created_date': created_date,
                'like_count': self.safe_get_field(row, 'like_count', '0'),
                'parent_comment_id': '0',  # USPE2024数据中没有这个字段
                'reply_comment': '',  # USPE2024数据中没有这个字段
                'location': '',  # USPE2024数据中没有location
                'stance': self.map_to_chinese(self.safe_get_field(row, 'stance'), self.stance_mapping),
                'sentiment': self.map_to_chinese(self.safe_get_field(row, 'sentiment'), self.sentiment_mapping),
                'intent': self.map_to_chinese(self.safe_get_field(row, 'intent'), self.intent_mapping),
                'stance_content': self.safe_get_field(row, 'stance_content', ''),
                'sentiment_content': self.safe_get_field(row, 'sentiment_content'),
                'intent_content': self.safe_get_field(row, 'intent_content'),
                'video_urls': self.safe_get_field(row, 'video_urls'),
                'img_urls': self.safe_get_field(row, 'img_urls')
            }

            processed_data.append(comment_data)

        result_df = pd.DataFrame(processed_data)
        logger.info(f"处理后评论数据: {len(result_df)} 条记录")

        return result_df

    def sort_data_by_time(self, df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
        """按时间排序数据"""
        if df.empty:
            return df

        try:
            # 转换时间列为数值类型进行排序
            df_copy = df.copy()
            df_copy['created_time_numeric'] = pd.to_numeric(df_copy['created_time'], errors='coerce')

            # 按时间排序
            df_sorted = df_copy.sort_values('created_time_numeric', ascending=ascending)

            # 将无效时间的行放到最后
            valid_time_mask = ~df_sorted['created_time_numeric'].isna()
            df_valid = df_sorted[valid_time_mask]
            df_invalid = df_sorted[~valid_time_mask]
            df_sorted = pd.concat([df_valid, df_invalid], ignore_index=True)

            # 删除临时排序列
            df_sorted = df_sorted.drop(columns=['created_time_numeric'])

            logger.info(f"数据已按时间排序完成")
            return df_sorted

        except Exception as e:
            logger.warning(f"时间排序失败: {e}，返回原始数据")
            return df

    def save_integrated_data(self, articles_df: pd.DataFrame, comments_df: pd.DataFrame,
                             output_folder: Path, candidate_name: str):
        """保存整合后的数据"""
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        # 保存文章数据
        if not articles_df.empty:
            # 按时间排序
            articles_df = self.sort_data_by_time(articles_df)

            articles_file = output_folder / f"USPE2024_{candidate_name}_integrated_articles.csv"
            articles_df.to_csv(articles_file, index=False, encoding='utf-8-sig')
            logger.info(f"文章数据已保存到: {articles_file}")

        # 保存评论数据
        if not comments_df.empty:
            # 按时间排序
            comments_df = self.sort_data_by_time(comments_df)

            comments_file = output_folder / f"USPE2024_{candidate_name}_integrated_comments.csv"
            comments_df.to_csv(comments_file, index=False, encoding='utf-8-sig')
            logger.info(f"评论数据已保存到: {comments_file}")

        # 保存ID映射关系
        self.save_id_mapping(output_folder, candidate_name)

        # 生成数据统计报告
        self.generate_data_report(articles_df, comments_df, output_folder, candidate_name)

    def save_id_mapping(self, output_folder: Path, candidate_name: str):
        """保存article_id到post_id的映射关系"""
        if self.article_id_mapping:
            mapping_file = output_folder / f"USPE2024_{candidate_name}_id_mapping.json"
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.article_id_mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"ID映射关系已保存到: {mapping_file}")

    def generate_data_report(self, articles_df: pd.DataFrame, comments_df: pd.DataFrame,
                             output_folder: Path, candidate_name: str):
        """生成数据统计报告"""
        report = {
            "USPE2024数据整合报告": {
                "处理时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "候选人": candidate_name,
                "数据来源": "USPE2024数据集",
                "处理说明": {
                    "多模态整合": "已将按多模态URL拆分的记录重新整合",
                    "英文映射": "已将英文的stance、sentiment、intent映射为中文",
                    "格式统一": "已转换为与XMSU7D数据集一致的格式"
                },
                "ID转换": {
                    "转换说明": "原始article_id已转换为统一的post_id格式",
                    "映射数量": len(self.article_id_mapping),
                    "示例映射": dict(list(self.article_id_mapping.items())[:5]) if self.article_id_mapping else {}
                },
                "文章数据": {
                    "总数": len(articles_df) if not articles_df.empty else 0,
                    "立场分析": articles_df['stance'].value_counts().to_dict() if not articles_df.empty else {},
                    "情感分析": articles_df['sentiment'].value_counts().to_dict() if not articles_df.empty else {},
                    "意图分析": articles_df['intent'].value_counts().to_dict() if not articles_df.empty else {}
                },
                "评论数据": {
                    "总数": len(comments_df) if not comments_df.empty else 0,
                    "立场分析": comments_df['stance'].value_counts().to_dict() if not comments_df.empty else {},
                    "情感分析": comments_df['sentiment'].value_counts().to_dict() if not comments_df.empty else {},
                    "意图分析": comments_df['intent'].value_counts().to_dict() if not comments_df.empty else {}
                }
            }
        }

        report_file = output_folder / f"USPE2024_{candidate_name}_data_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"数据报告已保存到: {report_file}")

    def create_unified_dataset(self, output_folder: Path, harris_articles: pd.DataFrame,
                               harris_comments: pd.DataFrame, trump_articles: pd.DataFrame,
                               trump_comments: pd.DataFrame):
        """创建统一的数据集，以支持Trump为主视角"""
        logger.info("正在创建统一数据集...")

        # 合并所有文章数据
        all_articles = []

        # 添加Trump文章（标记为支持Trump）
        if not trump_articles.empty:
            trump_articles_copy = trump_articles.copy()
            trump_articles_copy['target_candidate'] = 'Trump'
            trump_articles_copy['original_stance'] = trump_articles_copy['stance']  # 保存原始立场
            trump_articles_copy['stance'] = '支持'  # Trump本人发布的都是支持Trump的
            trump_articles_copy['data_source'] = 'Trump_Twitter'
            all_articles.append(trump_articles_copy)
            logger.info(f"添加Trump文章: {len(trump_articles_copy)} 条")

        # 添加Harris文章（标记为反对Trump，因为是竞争对手）
        if not harris_articles.empty:
            harris_articles_copy = harris_articles.copy()
            harris_articles_copy['target_candidate'] = 'Trump'
            harris_articles_copy['original_stance'] = harris_articles_copy['stance']  # 保存原始立场
            harris_articles_copy['stance'] = '反对'  # Harris作为竞争对手，立场是反对Trump的
            harris_articles_copy['data_source'] = 'Harris_Twitter'
            all_articles.append(harris_articles_copy)
            logger.info(f"添加Harris文章: {len(harris_articles_copy)} 条")

        # 合并所有评论数据
        all_comments = []

        # 添加Trump相关评论（保持原有立场）
        if not trump_comments.empty:
            trump_comments_copy = trump_comments.copy()
            trump_comments_copy['target_candidate'] = 'Trump'
            # 保存原始立场，并将原有的stance作为对Trump的立场
            trump_comments_copy['original_stance'] = trump_comments_copy['stance']
            # Trump相关评论的stance保持不变，因为它们已经是对Trump的立场
            trump_comments_copy['data_source'] = 'Trump_Comments'
            all_comments.append(trump_comments_copy)
            logger.info(f"添加Trump评论: {len(trump_comments_copy)} 条")

        # 添加Harris相关评论（需要转换立场视角）
        if not harris_comments.empty:
            harris_comments_copy = harris_comments.copy()
            harris_comments_copy['target_candidate'] = 'Trump'
            # 保存原始立场
            harris_comments_copy['original_stance'] = harris_comments_copy['stance']
            # 转换立场：支持Harris = 反对Trump，反对Harris = 支持Trump
            stance_conversion = {'支持': '反对', '反对': '支持', '中立': '中立'}
            harris_comments_copy['stance'] = harris_comments_copy['stance'].map(stance_conversion)
            harris_comments_copy['data_source'] = 'Harris_Comments'
            all_comments.append(harris_comments_copy)
            logger.info(f"添加Harris评论（立场已转换）: {len(harris_comments_copy)} 条")

        # 合并数据框
        if all_articles:
            unified_articles = pd.concat(all_articles, ignore_index=True)
            # 重新排序列，确保新字段在合适位置
            column_order = ['platform', 'post_id', 'original_article_id', 'target_candidate', 'stance',
                            'original_stance', 'data_source', 'content', 'created_time', 'created_date', 'like_count',
                            'comment_count', 'share_count', 'uid', 'username', 'location', 'sentiment', 'intent',
                            'stance_content', 'sentiment_content', 'intent_content', 'video_urls', 'img_urls']
            unified_articles = unified_articles.reindex(columns=column_order)
        else:
            unified_articles = pd.DataFrame()

        if all_comments:
            unified_comments = pd.concat(all_comments, ignore_index=True)
            # 重新排序列
            comment_column_order = ['platform', 'comment_id', 'content', 'post_id', 'original_article_id',
                                    'target_candidate', 'stance', 'original_stance', 'data_source', 'created_time',
                                    'created_date', 'like_count', 'parent_comment_id', 'reply_comment', 'location',
                                    'sentiment', 'intent', 'stance_content', 'sentiment_content',
                                    'intent_content', 'video_urls', 'img_urls']
            unified_comments = unified_comments.reindex(columns=comment_column_order)
        else:
            unified_comments = pd.DataFrame()

        # 按时间排序
        if not unified_articles.empty:
            unified_articles = self.sort_data_by_time(unified_articles)
        if not unified_comments.empty:
            unified_comments = self.sort_data_by_time(unified_comments)

        # 保存统一数据集
        self.save_unified_dataset(unified_articles, unified_comments, output_folder)

        return unified_articles, unified_comments

    def save_unified_dataset(self, articles_df: pd.DataFrame, comments_df: pd.DataFrame, output_folder: Path):
        """保存统一数据集"""
        logger.info("保存统一数据集...")

        # 保存文章数据
        if not articles_df.empty:
            articles_file = output_folder / "USPE2024_unified_articles2.csv"
            articles_df.to_csv(articles_file, index=False, encoding='utf-8-sig')
            logger.info(f"统一文章数据已保存到: {articles_file}")

        # 保存评论数据
        if not comments_df.empty:
            comments_file = output_folder / "USPE2024_unified_comments2.csv"
            comments_df.to_csv(comments_file, index=False, encoding='utf-8-sig')
            logger.info(f"统一评论数据已保存到: {comments_file}")

        # 生成统一数据集报告
        self.generate_unified_report(articles_df, comments_df, output_folder)

    def generate_unified_report(self, articles_df: pd.DataFrame, comments_df: pd.DataFrame, output_folder: Path):
        """生成统一数据集报告"""
        report = {
            "USPE2024统一数据集报告": {
                "生成时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "数据说明": {
                    "视角": "以支持Trump为主视角的统一数据集",
                    "立场转换": "Harris相关数据的立场已转换为对Trump的立场",
                    "数据来源": "整合Harris和Trump的Twitter数据"
                },
                "统一文章数据": {
                    "总数": len(articles_df) if not articles_df.empty else 0,
                    "数据来源分布": articles_df['data_source'].value_counts().to_dict() if not articles_df.empty else {},
                    "对Trump立场分布": articles_df['stance'].value_counts().to_dict() if not articles_df.empty else {},
                    "原始立场分布": articles_df['original_stance'].value_counts().to_dict() if not articles_df.empty else {},
                    "情感分析": articles_df['sentiment'].value_counts().to_dict() if not articles_df.empty else {},
                    "意图分析": articles_df['intent'].value_counts().to_dict() if not articles_df.empty else {}
                },
                "统一评论数据": {
                    "总数": len(comments_df) if not comments_df.empty else 0,
                    "数据来源分布": comments_df['data_source'].value_counts().to_dict() if not comments_df.empty else {},
                    "对Trump立场分布": comments_df['stance'].value_counts().to_dict() if not comments_df.empty else {},
                    "原始立场分布": comments_df['original_stance'].value_counts().to_dict() if not comments_df.empty else {},
                    "情感分析": comments_df['sentiment'].value_counts().to_dict() if not comments_df.empty else {},
                    "意图分析": comments_df['intent'].value_counts().to_dict() if not comments_df.empty else {}
                },
                "时间范围": {
                    "文章时间范围": f"{articles_df['created_date'].min()} 至 {articles_df['created_date'].max()}" if not articles_df.empty else "无数据",
                    "评论时间范围": f"{comments_df['created_date'].min()} 至 {comments_df['created_date'].max()}" if not comments_df.empty else "无数据"
                }
            }
        }

        report_file = output_folder / "USPE2024_unified_data_report2.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"统一数据集报告已保存到: {report_file}")

    def run_integration(self, output_path: str = None):
        """运行数据整合流程"""
        if output_path is None:
            output_path = self.data_path / "integrated_data"

        output_path = Path(output_path)

        # 处理Harris数据
        logger.info("="*50)
        logger.info("开始处理Harris数据")
        logger.info("="*50)

        # 重置映射关系
        self.article_id_mapping = {}

        # 处理Harris文章数据
        harris_articles_file = self.data_path / "harris_article2.csv"
        if harris_articles_file.exists():
            harris_articles = self.process_article_data(harris_articles_file, "twitter_harris")
        else:
            harris_articles = pd.DataFrame()
            logger.warning("Harris文章数据文件不存在")

        # 处理Harris评论数据
        harris_comments_file = self.data_path / "harris_comments.csv"
        if harris_comments_file.exists():
            harris_comments = self.process_comment_data(harris_comments_file, "twitter_harris", harris_articles)
        else:
            harris_comments = pd.DataFrame()
            logger.warning("Harris评论数据文件不存在")

        # 保存Harris数据
        self.save_integrated_data(harris_articles, harris_comments, output_path, "Harris")

        # 处理Trump数据
        logger.info("="*50)
        logger.info("开始处理Trump数据")
        logger.info("="*50)

        # 重置映射关系
        self.article_id_mapping = {}

        # 处理Trump文章数据
        trump_articles_file = self.data_path / "trump_article2.csv"
        if trump_articles_file.exists():
            trump_articles = self.process_article_data(trump_articles_file, "twitter_trump")
        else:
            trump_articles = pd.DataFrame()
            logger.warning("Trump文章数据文件不存在")

        # 处理Trump评论数据
        trump_comments_file = self.data_path / "trump_comments.csv"
        if trump_comments_file.exists():
            trump_comments = self.process_comment_data(trump_comments_file, "twitter_trump", trump_articles)
        else:
            trump_comments = pd.DataFrame()
            logger.warning("Trump评论数据文件不存在")

        # 保存Trump数据
        self.save_integrated_data(trump_articles, trump_comments, output_path, "Trump")

        # 整合所有数据为统一数据集（以支持Trump为主视角）
        logger.info("="*50)
        logger.info("开始整合所有数据为统一数据集")
        logger.info("="*50)

        self.create_unified_dataset(output_path, harris_articles, harris_comments, trump_articles, trump_comments)

        logger.info("="*50)
        logger.info("USPE2024数据整合完成！")
        logger.info(f"输出目录: {output_path}")
        logger.info("="*50)


def main():
    """主函数"""
    # 设置数据路径
    data_path = "Data/USPE2024"

    # 创建数据整合器
    organizer = USPE2024DataOrganizer(data_path)

    # 运行数据整合
    organizer.run_integration()


if __name__ == "__main__":
    main()
